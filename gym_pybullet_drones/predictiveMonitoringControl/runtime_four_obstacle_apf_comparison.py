"""
Comparison of three navigation strategies in the SAME four-obstacle scenario:

1. Direct waypoint baseline
2. Classical reactive Artificial Potential Field (APF)
3. Proposed predictive runtime supervisor

The final runtime_four_obstacle.py is imported and left unchanged.

Run from the repository root, e.g.:

python -m gym_pybullet_drones.predictiveMonitoringControl.runtime_four_obstacle_apf_comparison

Outputs:
    apf_comparison_results/
        comparison_summary.csv
        apf_log.csv
        baseline_log.csv
        predictive_monitor_log.csv
        comparison_trajectory.png
        comparison_clearance.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import Physics
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary


# ==========================================================
# Import the FROZEN final implementation
# ==========================================================

try:
    import runtime_four_obstacle as base
except ModuleNotFoundError:
    from gym_pybullet_drones.predictiveMonitoringControl import (
        runtime_four_obstacle as base
    )


# ==========================================================
# Experiment settings
# ==========================================================

OUTPUT_FOLDER = "apf_comparison_results"

CONTROL_FREQ_HZ = 48
SIMULATION_FREQ_HZ = 240

# Enough time for every method to finish.
MAX_DURATION_SEC = 50

GOAL_REACH_DIST = 0.12


# ==========================================================
# APF configuration
# ==========================================================
#
# Classical reactive potential-field structure:
#
# F = F_att + F_rep
#
# F_att points toward the goal.
#
# For obstacle clearance d < d0:
#
# F_rep =
#     k_rep (1/d - 1/d0) (1/d^2) n
#
# where n points away from the obstacle.
#
# No lookahead prediction is used.
# ==========================================================

APF_K_ATT = 1.0

# Repulsive gain
APF_K_REP = 0.002

# Obstacle influence distance [m]
APF_INFLUENCE_DIST = 0.25

# Avoid numerical explosion very close to obstacle
APF_MIN_DISTANCE = 0.015

# Cap repulsive magnitude so the simulation stays well behaved
APF_MAX_REPULSION = 3.0

# Same nominal speed as the other controllers
APF_SPEED = 0.34


# ==========================================================
# Utilities
# ==========================================================

def nearest_clearance(pos_xy, obstacles):

    best_clearance = np.inf
    best_obs = None

    for obs in obstacles:

        obs_xy = np.array(
            [obs.cx, obs.cy],
            dtype=float
        )

        center_distance = np.linalg.norm(
            pos_xy - obs_xy
        )

        clearance = (
            center_distance - obs.radius
        )

        if clearance < best_clearance:

            best_clearance = clearance
            best_obs = obs

    return best_obs, float(best_clearance)


def velocity_to_command(vel_xy, speed):

    norm = np.linalg.norm(vel_xy)

    if norm < 1e-12:

        return np.array(
            [0.0, 0.0, 0.0, 0.0]
        )

    direction = vel_xy / norm

    return np.array([
        direction[0],
        direction[1],
        0.0,
        speed
    ])


# ==========================================================
# APF controller
# ==========================================================

class APFController:

    def __init__(
        self,
        obstacles,
        goal_xy,
        k_att=APF_K_ATT,
        k_rep=APF_K_REP,
        influence_dist=APF_INFLUENCE_DIST,
        speed=APF_SPEED,
    ):

        self.obstacles = obstacles

        self.goal_xy = np.array(
            goal_xy,
            dtype=float
        )

        self.k_att = k_att
        self.k_rep = k_rep
        self.influence_dist = influence_dist
        self.speed = speed


    def attractive_force(self, pos_xy):

        to_goal = (
            self.goal_xy - pos_xy
        )

        distance = np.linalg.norm(
            to_goal
        )

        if distance < 1e-12:

            return np.zeros(2)

        # Unit attractive direction.
        return (
            self.k_att
            * to_goal / distance
        )


    def repulsive_force(self, pos_xy):

        total_repulsion = np.zeros(2)

        active_obstacles = 0

        for obs in self.obstacles:

            obs_xy = np.array([
                obs.cx,
                obs.cy
            ])

            away = (
                pos_xy - obs_xy
            )

            center_distance = np.linalg.norm(
                away
            )

            if center_distance < 1e-12:
                continue

            clearance = (
                center_distance
                - obs.radius
            )

            # Classical APF acts only inside
            # the influence distance.
            if clearance >= self.influence_dist:
                continue

            active_obstacles += 1

            away_dir = (
                away / center_distance
            )

            # Protect against d <= 0 and numerical explosion.
            d = max(
                clearance,
                APF_MIN_DISTANCE
            )

            magnitude = (
                self.k_rep
                * (
                    (1.0 / d)
                    -
                    (1.0 / self.influence_dist)
                )
                * (1.0 / (d * d))
            )

            magnitude = min(
                magnitude,
                APF_MAX_REPULSION
            )

            total_repulsion += (
                magnitude * away_dir
            )

        return (
            total_repulsion,
            active_obstacles
        )


    def command(self, pos_xy):

        f_att = self.attractive_force(
            pos_xy
        )

        f_rep, active_obstacles = (
            self.repulsive_force(
                pos_xy
            )
        )

        total_force = (
            f_att + f_rep
        )

        norm = np.linalg.norm(
            total_force
        )

        if norm < 1e-12:

            # APF local minimum.
            cmd = np.array(
                [0.0, 0.0, 0.0, 0.0]
            )

            return (
                cmd,
                f_att,
                f_rep,
                active_obstacles,
                True
            )

        direction = (
            total_force / norm
        )

        cmd = np.array([
            direction[0],
            direction[1],
            0.0,
            self.speed
        ])

        return (
            cmd,
            f_att,
            f_rep,
            active_obstacles,
            False
        )


# ==========================================================
# Run APF experiment
# ==========================================================

def run_apf_case():

    init_xyzs = np.array([
        [-1.10, 0.12, 0.30]
    ])

    init_rpys = np.array([
        [0.0, 0.0, 0.0]
    ])

    env = VelocityAviary(
        drone_model=base.DEFAULT_DRONE,
        num_drones=1,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        physics=Physics.PYB,
        neighbourhood_radius=10,
        pyb_freq=SIMULATION_FREQ_HZ,
        ctrl_freq=CONTROL_FREQ_HZ,
        gui=False,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )

    obstacles = base.obstacle_map()

    # EXACT same final goal as the frozen experiment.
    goal_xy = np.array([
        1.90,
        -0.04
    ])

    apf = APFController(
        obstacles=obstacles,
        goal_xy=goal_xy
    )

    action = np.zeros(
        (1, 4)
    )

    max_steps = int(
        MAX_DURATION_SEC
        * CONTROL_FREQ_HZ
    )

    logs = []

    reached_goal = False
    completion_time = np.nan

    local_minimum_steps = 0

    for i in range(max_steps):

        t = (
            i / CONTROL_FREQ_HZ
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        pos = base.extract_position(
            obs[0]
        )

        pos_xy = np.array(
            pos[:2],
            dtype=float
        )

        (
            cmd,
            f_att,
            f_rep,
            active_obstacles,
            local_minimum,
        ) = apf.command(pos_xy)

        if local_minimum:
            local_minimum_steps += 1

        action[0, :] = cmd

        nearest_obs, clearance = (
            nearest_clearance(
                pos_xy,
                obstacles
            )
        )

        speed = base.command_speed(
            cmd
        )

        risk_score = max(
            0.0,
            base.SupervisorConfig().safe_clearance
            - clearance
        )

        logs.append({

            "time_s": t,

            "x_m": pos[0],
            "y_m": pos[1],
            "z_m": pos[2],

            "current_clearance_m":
                clearance,

            "nearest_obstacle":
                (
                    nearest_obs.name
                    if nearest_obs is not None
                    else "none"
                ),

            "executed_speed_mps":
                speed,

            "risk_score":
                risk_score,

            "apf_attractive_x":
                f_att[0],

            "apf_attractive_y":
                f_att[1],

            "apf_repulsive_x":
                f_rep[0],

            "apf_repulsive_y":
                f_rep[1],

            "apf_active_obstacles":
                active_obstacles,

            "local_minimum":
                int(local_minimum),
        })

        # Same goal-completion criterion.
        if np.linalg.norm(
            pos_xy - goal_xy
        ) < GOAL_REACH_DIST:

            reached_goal = True
            completion_time = t

            print(
                f"APF reached final goal "
                f"at t={t:.2f} s"
            )

            break

    env.close()

    df = pd.DataFrame(
        logs
    )

    return (
        df,
        obstacles,
        reached_goal,
        completion_time,
        local_minimum_steps,
    )


# ==========================================================
# Common summary
# ==========================================================

def summarize_case(
    df,
    method,
    goal_xy,
):

    final_xy = np.array([
        df["x_m"].iloc[-1],
        df["y_m"].iloc[-1]
    ])

    final_goal_distance = float(
        np.linalg.norm(
            final_xy - goal_xy
        )
    )

    reached_goal = (
        final_goal_distance
        < GOAL_REACH_DIST
    )

    completion_time = (
        float(df["time_s"].iloc[-1])
        if reached_goal
        else np.nan
    )

    return {

        "method":
            method,

        "reached_goal":
            int(reached_goal),

        "completion_time_s":
            completion_time,

        "final_goal_error_m":
            final_goal_distance,

        "min_clearance_m":
            float(
                df[
                    "current_clearance_m"
                ].min()
            ),

        "collision_steps":
            int(
                (
                    df[
                        "current_clearance_m"
                    ] < 0.0
                ).sum()
            ),

        "close_steps_less_8cm":
            int(
                (
                    df[
                        "current_clearance_m"
                    ] < 0.08
                ).sum()
            ),

        "mean_speed_mps":
            float(
                df[
                    "executed_speed_mps"
                ].mean()
            ),

        "min_speed_mps":
            float(
                df[
                    "executed_speed_mps"
                ].min()
            ),

        "max_risk_score":
            float(
                df[
                    "risk_score"
                ].max()
            )
            if "risk_score" in df.columns
            else np.nan,
    }


# ==========================================================
# Main comparison
# ==========================================================

def main():

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    goal_xy = np.array([
        1.90,
        -0.04
    ])

    # ======================================================
    # 1. DIRECT BASELINE
    # ======================================================

    print(
        "\n======================================"
    )

    print(
        "Running direct waypoint baseline..."
    )

    print(
        "======================================"
    )

    df_base, cfg, obstacles = (
        base.run_case(
            use_supervisor=False,
            gui=False,
            plot=False,
            duration_sec=MAX_DURATION_SEC,
            simulation_freq_hz=SIMULATION_FREQ_HZ,
            control_freq_hz=CONTROL_FREQ_HZ,
            record_video=False,
            user_debug_gui=False,
            obstacles=False,
        )
    )

    # ======================================================
    # 2. APF
    # ======================================================

    print(
        "\n======================================"
    )

    print(
        "Running reactive APF..."
    )

    print(
        "======================================"
    )

    (
        df_apf,
        _,
        apf_reached_goal,
        apf_completion_time,
        local_minimum_steps,
    ) = run_apf_case()

    # ======================================================
    # 3. PROPOSED PREDICTIVE MONITOR
    # ======================================================

    print(
        "\n======================================"
    )

    print(
        "Running predictive runtime monitor..."
    )

    print(
        "======================================"
    )

    df_pred, cfg, obstacles = (
        base.run_case(
            use_supervisor=True,
            gui=False,
            plot=False,
            duration_sec=MAX_DURATION_SEC,
            simulation_freq_hz=SIMULATION_FREQ_HZ,
            control_freq_hz=CONTROL_FREQ_HZ,
            record_video=False,
            user_debug_gui=False,
            obstacles=False,
        )
    )

    # ======================================================
    # Save logs
    # ======================================================

    df_base.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "baseline_log.csv"
        ),
        index=False
    )

    df_apf.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "apf_log.csv"
        ),
        index=False
    )

    df_pred.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "predictive_monitor_log.csv"
        ),
        index=False
    )

    # ======================================================
    # Summary
    # ======================================================

    summaries = [

        summarize_case(
            df_base,
            "Direct baseline",
            goal_xy
        ),

        summarize_case(
            df_apf,
            "Reactive APF",
            goal_xy
        ),

        summarize_case(
            df_pred,
            "Predictive monitor",
            goal_xy
        ),
    ]

    summary = pd.DataFrame(
        summaries
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "comparison_summary.csv"
        ),
        index=False
    )

    print(
        "\n============================================"
    )

    print(
        "THREE-METHOD COMPARISON"
    )

    print(
        "============================================\n"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nAPF local-minimum steps:",
        local_minimum_steps
    )

    # ======================================================
    # Trajectory comparison
    # ======================================================

    plt.figure(figsize=(7.2, 4.3))

    # ------------------------------------------------------
    # Trajectories
    # ------------------------------------------------------

    plt.plot(
        df_base["x_m"],
        df_base["y_m"],
        linewidth=2,
        label="Direct baseline"
    )

    plt.plot(
        df_apf["x_m"],
        df_apf["y_m"],
        linewidth=2,
        label="Reactive APF"
    )

    plt.plot(
        df_pred["x_m"],
        df_pred["y_m"],
        linewidth=2,
        label="Predictive monitor"
    )

    ax = plt.gca()

    # ------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------

    for obs in obstacles:

        circle = plt.Circle(
            (obs.cx, obs.cy),
            obs.radius,
            fill=False,
            linestyle="--",
            linewidth=2,
            color="black"
        )

        ax.add_patch(circle)

        ax.text(
            obs.cx,
            obs.cy,
            obs.name,
            ha="center",
            va="center"
        )

    # ------------------------------------------------------
    # Goal
    # ------------------------------------------------------

    plt.scatter(
        goal_xy[0],
        goal_xy[1],
        marker="*",
        s=120,
        label="Goal"
    )

    # ------------------------------------------------------
    # Flight boundary
    # Same boundary used in the previous supervisor figure
    # ------------------------------------------------------

    x_min = -1.40
    x_max = 2.05
    y_min = -0.70
    y_max = 0.70

    boundary_x = [
        x_min, x_max, x_max, x_min, x_min
    ]

    boundary_y = [
        y_min, y_min, y_max, y_max, y_min
    ]

    plt.plot(
        boundary_x,
        boundary_y,
        linestyle="--",
        linewidth=1.5,
        label="Flight boundary"
    )

    # ------------------------------------------------------
    # Figure formatting
    # ------------------------------------------------------

    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")

    plt.axis("equal")

    # Slightly larger limits so the dashed boundary is visible
    # rather than overlapping the axes.
    plt.xlim(-1.50, 2.15)
    plt.ylim(-0.80, 0.80)

    plt.grid(True)

    plt.legend(
        fontsize=8,
        loc="upper right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "comparison_trajectory.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================================
    # Clearance comparison
    # ======================================================

    plt.figure(
        figsize=(8, 4)
    )

    plt.plot(
        df_base["time_s"],
        df_base["current_clearance_m"],
        linewidth=2,
        label="Direct baseline"
    )

    plt.plot(
        df_apf["time_s"],
        df_apf["current_clearance_m"],
        linewidth=2,
        label="Reactive APF"
    )

    plt.plot(
        df_pred["time_s"],
        df_pred["current_clearance_m"],
        linewidth=2,
        label="Predictive monitor"
    )

    plt.axhline(
        0.0,
        linestyle="--",
        label="Collision boundary"
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Clearance [m]"
    )

    plt.grid(
        True
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "comparison_clearance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSaved outputs in:"
    )

    print(
        OUTPUT_FOLDER
    )


if __name__ == "__main__":
    main()
