# Save this file as:
# gym_pybullet_drones/predictiveMonitoringControl/run_camera_oracle_sensitivity.py
#
# Run from repository root:
# python -m gym_pybullet_drones.predictiveMonitoringControl.run_camera_oracle_sensitivity
#
# Optional:
# python -m gym_pybullet_drones.predictiveMonitoringControl.run_camera_oracle_sensitivity --duration_sec 180 --output_folder results/camera_oracle_sensitivity

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from .perception import ColorObstaclePerception, PerceptionConfig
from .safety_oracle import CameraSafetyOracle, SafetyOracleConfig
from .state_estimator import CameraStateEstimator, StateEstimatorConfig
from .virtual_camera import CameraConfig, VirtualCamera


DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 180
DEFAULT_OUTPUT_FOLDER = "results/camera_oracle_sensitivity"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

START = np.array([-1.20, 0.00], dtype=float)
GOAL = np.array([4.20, 0.00], dtype=float)
GOAL_TOLERANCE = 0.12

SAFETY_MARGIN = 0.18

OBSTACLE_DATA = [
    ("O1", -0.10,  0.18, 0.20, [1.0, 0.0, 0.0, 1.0]),
    ("O2",  1.15, -0.18, 0.20, [0.0, 0.0, 1.0, 1.0]),
    ("O3",  2.40,  0.18, 0.20, [1.0, 1.0, 0.0, 1.0]),
    ("O4",  3.65, -0.18, 0.20, [0.0, 1.0, 0.0, 1.0]),
]


@dataclass(frozen=True)
class SensitivitySetting:
    name: str
    warning_distance_m: float
    critical_distance_m: float
    nominal_speed: float
    avoid_speed: float
    slow_speed: float
    lateral_command_strength: float
    critical_lateral_strength: float


# Nominal values are taken directly from the current experiment code.
NOMINAL = SensitivitySetting(
    name="nominal",
    warning_distance_m=1.20,
    critical_distance_m=0.65,
    nominal_speed=0.18,
    avoid_speed=0.17,
    slow_speed=0.13,
    lateral_command_strength=1.45,
    critical_lateral_strength=1.70,
)


def scaled_setting(
    name: str,
    threshold_scale: float = 1.0,
    speed_scale: float = 1.0,
    lateral_scale: float = 1.0,
) -> SensitivitySetting:
    """One-factor-at-a-time perturbation around the nominal setting."""
    return SensitivitySetting(
        name=name,
        warning_distance_m=NOMINAL.warning_distance_m * threshold_scale,
        critical_distance_m=NOMINAL.critical_distance_m * threshold_scale,
        nominal_speed=NOMINAL.nominal_speed,  # keep mission nominal speed fixed
        avoid_speed=NOMINAL.avoid_speed * speed_scale,
        slow_speed=NOMINAL.slow_speed * speed_scale,
        lateral_command_strength=NOMINAL.lateral_command_strength * lateral_scale,
        critical_lateral_strength=NOMINAL.critical_lateral_strength * lateral_scale,
    )


SETTINGS = [
    NOMINAL,
    scaled_setting("threshold_minus_10pct", threshold_scale=0.90),
    scaled_setting("threshold_plus_10pct", threshold_scale=1.10),
    scaled_setting("speed_minus_10pct", speed_scale=0.90),
    scaled_setting("speed_plus_10pct", speed_scale=1.10),
    scaled_setting("lateral_minus_10pct", lateral_scale=0.90),
    scaled_setting("lateral_plus_10pct", lateral_scale=1.10),
]


def normalize_direction(vector_xy: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector_xy))
    if norm < 1e-9:
        return np.array([0.0, 0.0], dtype=float)
    return np.asarray(vector_xy, dtype=float) / norm


def create_visible_obstacles(client_id: int) -> list[int]:
    obstacle_height = 0.50
    ids: list[int] = []

    for _, x, y, radius, colour in OBSTACLE_DATA:
        collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            height=obstacle_height,
            physicsClientId=client_id,
        )
        visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=obstacle_height,
            rgbaColor=colour,
            physicsClientId=client_id,
        )
        body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, obstacle_height / 2.0],
            physicsClientId=client_id,
        )
        ids.append(body)

    return ids


def calculate_clearance(position_xy: np.ndarray) -> tuple[float, str]:
    min_clearance = float("inf")
    nearest = "none"

    for name, x, y, radius, _ in OBSTACLE_DATA:
        center = np.array([x, y], dtype=float)
        clearance = float(np.linalg.norm(position_xy - center) - radius)

        if clearance < min_clearance:
            min_clearance = clearance
            nearest = name

    return min_clearance, nearest


def save_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one_setting(
    setting: SensitivitySetting,
    duration_sec: int,
    simulation_freq_hz: int,
    control_freq_hz: int,
    output_directory: str,
) -> dict:
    print("\n" + "=" * 72)
    print(f"Running sensitivity setting: {setting.name}")
    print(asdict(setting))
    print("=" * 72)

    env = VelocityAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[START[0], START[1], 0.30]], dtype=float),
        initial_rpys=np.array([[0.0, 0.0, 0.0]], dtype=float),
        physics=Physics.PYB,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=False,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )

    create_visible_obstacles(env.CLIENT)

    camera = VirtualCamera(
        client_id=env.CLIENT,
        drone_body_id=env.DRONE_IDS[0],
        config=CameraConfig(
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fov_deg=70.0,
            near_plane=0.02,
            far_plane=5.0,
            offset_body=(0.05, 0.0, 0.025),
            pitch_deg=-8.0,
        ),
    )

    perception = ColorObstaclePerception(
        config=PerceptionConfig(
            image_width=CAMERA_WIDTH,
            image_height=CAMERA_HEIGHT,
            horizontal_fov_deg=70.0,
            obstacle_diameter_m=0.40,
            minimum_contour_area_px=300.0,
            minimum_width_px=10.0,
            morphology_kernel_size=5,
        )
    )

    state_estimator = CameraStateEstimator(
        config=StateEstimatorConfig(
            minimum_confidence=0.15,
            maximum_distance_m=5.0,
            maximum_frame_age_s=0.50,
        )
    )

    oracle = CameraSafetyOracle(
        config=SafetyOracleConfig(
            nominal_speed=setting.nominal_speed,
            avoid_speed=setting.avoid_speed,
            slow_speed=setting.slow_speed,
            warning_distance_m=setting.warning_distance_m,
            critical_distance_m=setting.critical_distance_m,
            lateral_command_strength=setting.lateral_command_strength,
            critical_lateral_strength=setting.critical_lateral_strength,
            minimum_confidence=0.15,
            maximum_frame_age_s=0.50,
            release_after_missing_frames=8,
        )
    )

    initial_direction = normalize_direction(GOAL - START)
    action = np.array(
        [[initial_direction[0], initial_direction[1], 0.0, setting.nominal_speed]],
        dtype=float,
    )

    observations = []
    estimated_obstacles = []
    decision = oracle.evaluate(
        nominal_direction_xy=initial_direction,
        estimated_obstacles=[],
    )

    number_of_steps = int(duration_sec * env.CTRL_FREQ)
    camera_interval_steps = 3

    trajectory_rows: list[dict] = []

    previous_xy: np.ndarray | None = None
    path_length = 0.0
    min_clearance = float("inf")
    max_risk = 0.0
    intervention_steps = 0
    total_steps = 0
    reached_goal = False
    collided = False
    end_time_s = 0.0

    try:
        for step in range(number_of_steps):
            observation, reward, terminated, truncated, info = env.step(action)

            simulation_time = step / env.CTRL_FREQ
            end_time_s = simulation_time

            drone_position = np.asarray(observation[0][0:3], dtype=float)
            position_xy = drone_position[:2]

            if previous_xy is not None:
                path_length += float(np.linalg.norm(position_xy - previous_xy))
            previous_xy = position_xy.copy()

            physical_clearance, nearest_true = calculate_clearance(position_xy)
            min_clearance = min(min_clearance, physical_clearance)

            if physical_clearance <= 0.0:
                collided = True

            to_goal = GOAL - position_xy
            goal_distance = float(np.linalg.norm(to_goal))
            nominal_direction_xy = normalize_direction(to_goal)

            if step % camera_interval_steps == 0:
                rgb_frame = camera.capture()["rgb"]

                observations = perception.process(
                    rgb_frame=rgb_frame,
                    timestamp_s=simulation_time,
                )

                estimated_obstacles = state_estimator.update(
                    observations=observations,
                    current_time_s=simulation_time,
                )

                decision = oracle.evaluate(
                    nominal_direction_xy=nominal_direction_xy,
                    estimated_obstacles=estimated_obstacles,
                )

                action[0, :] = decision.safe_command

            total_steps += 1

            if decision.mode != "NOMINAL":
                intervention_steps += 1

            max_risk = max(max_risk, float(decision.risk_score))

            trajectory_rows.append(
                {
                    "time_s": f"{simulation_time:.6f}",
                    "x_m": f"{drone_position[0]:.6f}",
                    "y_m": f"{drone_position[1]:.6f}",
                    "z_m": f"{drone_position[2]:.6f}",
                    "physical_clearance_m": f"{physical_clearance:.6f}",
                    "nearest_true_obstacle": nearest_true,
                    "control_mode": decision.mode,
                    "risk_score": f"{decision.risk_score:.6f}",
                    "commanded_speed_mps": f"{decision.safe_command[3]:.6f}",
                }
            )

            if goal_distance < GOAL_TOLERANCE:
                reached_goal = True
                break

            if terminated or truncated:
                break

    finally:
        env.close()

    intervention_fraction = (
        intervention_steps / total_steps if total_steps > 0 else float("nan")
    )

    setting_dir = os.path.join(output_directory, setting.name)
    os.makedirs(setting_dir, exist_ok=True)

    save_rows(
        os.path.join(setting_dir, "trajectory.csv"),
        trajectory_rows,
    )

    result = {
        "setting": setting.name,
        "warning_distance_m": setting.warning_distance_m,
        "critical_distance_m": setting.critical_distance_m,
        "nominal_speed_mps": setting.nominal_speed,
        "avoid_speed_mps": setting.avoid_speed,
        "critical_speed_mps": setting.slow_speed,
        "lateral_weight": setting.lateral_command_strength,
        "critical_lateral_weight": setting.critical_lateral_strength,
        "min_clearance_m": min_clearance,
        "path_length_m": path_length,
        "mission_time_s": end_time_s,
        "intervention_fraction": intervention_fraction,
        "max_risk": max_risk,
        "reached_goal": int(reached_goal),
        "collision": int(collided),
    }

    print(
        f"{setting.name}: "
        f"c_min={min_clearance:.3f} m, "
        f"L={path_length:.3f} m, "
        f"T={end_time_s:.2f} s, "
        f"eta_int={intervention_fraction:.3f}, "
        f"r_max={max_risk:.3f}, "
        f"goal={reached_goal}, collision={collided}"
    )

    return result


def print_paper_summary(results: list[dict]) -> None:
    print("\n\n" + "#" * 72)
    print("SENSITIVITY SUMMARY")
    print("#" * 72)

    header = (
        f"{'Setting':28s} "
        f"{'c_min [m]':>10s} "
        f"{'T [s]':>10s} "
        f"{'L [m]':>10s} "
        f"{'eta_int':>10s}"
    )
    print(header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row['setting']:28s} "
            f"{row['min_clearance_m']:10.3f} "
            f"{row['mission_time_s']:10.2f} "
            f"{row['path_length_m']:10.3f} "
            f"{row['intervention_fraction']:10.3f}"
        )

    nominal = next(row for row in results if row["setting"] == "nominal")

    print("\nNominal reference:")
    print(
        f"c_min={nominal['min_clearance_m']:.3f} m, "
        f"T={nominal['mission_time_s']:.2f} s"
    )
    print(
        "\nUse the generated sensitivity_summary.csv for the camera-ready "
        "sensitivity paragraph/table."
    )


def main(
    duration_sec: int = DEFAULT_DURATION_SEC,
    simulation_freq_hz: int = DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz: int = DEFAULT_CONTROL_FREQ_HZ,
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = os.path.join(
        output_folder,
        f"sensitivity_{timestamp}",
    )
    os.makedirs(output_directory, exist_ok=True)

    results = []

    for setting in SETTINGS:
        result = run_one_setting(
            setting=setting,
            duration_sec=duration_sec,
            simulation_freq_hz=simulation_freq_hz,
            control_freq_hz=control_freq_hz,
            output_directory=output_directory,
        )
        results.append(result)

    summary_path = os.path.join(
        output_directory,
        "sensitivity_summary.csv",
    )
    save_rows(summary_path, results)

    print_paper_summary(results)

    print(f"\nSaved summary CSV: {summary_path}")
    print(f"Saved per-setting trajectories under: {output_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "One-factor-at-a-time sensitivity analysis for the "
            "closed-loop camera safety oracle."
        )
    )

    parser.add_argument(
        "--duration_sec",
        default=DEFAULT_DURATION_SEC,
        type=int,
    )
    parser.add_argument(
        "--simulation_freq_hz",
        default=DEFAULT_SIMULATION_FREQ_HZ,
        type=int,
    )
    parser.add_argument(
        "--control_freq_hz",
        default=DEFAULT_CONTROL_FREQ_HZ,
        type=int,
    )
    parser.add_argument(
        "--output_folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
    )

    args = parser.parse_args()
    main(**vars(args))
