import optuna
import numpy as np
import math
import time
import pinocchio as pin
from joint_interpolator import SegmentedJointTrap
from sharework import loadSharework
from fake_command_bridge import FakeCommandBridge
from optimal_cbf_task_controller import BCFOptimalController, ControllerConfig

# ----------------- STATIC INITIALIZATION (only once) -----------------
UR10E_JOINTS = [
    "ur10e_shoulder_pan_joint",
    "ur10e_shoulder_lift_joint",
    "ur10e_elbow_joint",
    "ur10e_wrist_1_joint",
    "ur10e_wrist_2_joint",
    "ur10e_wrist_3_joint",
]
model_wrapper = loadSharework(UR10E_JOINTS)

# Camera and bridge
R = pin.utils.rotate('z', 1.9) @ pin.utils.rotate('x', 1.57)
T_wc = pin.SE3(R, np.array([-1.85, -0.9, 0.9]))
bridge = FakeCommandBridge(
    UR10E_JOINTS,
    csv_path="a01_s10_e02_skeleton3D_with_savgol_vel_acc.csv",
    Tworld_to_cam=T_wc,
    slowdown_factor=0.4,
)

home = np.array([90, -140, 140, -90, 90, 0]) * np.pi / 180.0
Tc =2e-3
gen_cfg = ControllerConfig(Tc=Tc)
# Basic planner reused across trials
planner = SegmentedJointTrap(Dq_max=gen_cfg.Dq_max*.3, DDq_max=gen_cfg.DDq_max*.3)
planner.addWayPoint(home)
planner.addWayPoint(home + np.deg2rad([0, 20, -20, 0, 0, 0]))
planner.addWayPoint(home)
T_total = planner.computeTime()

# -------------------- EVALUATION FUNCTION --------------------
def run_episode(lambda1, lambda2, lambda3, lambda4, Tc=2e-3, duration=5.0):
    cfg = ControllerConfig(Tc=Tc)
    cfg.lambda1, cfg.lambda2, cfg.lambda3, cfg.lambda4 = lambda1, lambda2, lambda3, lambda4
    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg)

    q = home.copy()
    ctrl.reset_state(q)
    t = 0.0
    violations, nsteps = 0, 0
    sum_scale = 0.0

    while t < duration:
        obs_pos, obs_vel, obs_acc = bridge.getObstacles()
        nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(t % T_total)

        try:
            out = ctrl.step(
                obs_pos=obs_pos,
                obs_vel=obs_vel,
                obs_acc=obs_acc,
                nominal_q=nominal_q,
                nominal_Dq=nominal_Dq,
                nominal_DDq=nominal_DDq,
            )
        except Exception:
            # Penalize infeasible or divergent QP
            return 1.0, -1.0

        if out["h_min"] < 0:
            violations += 1
        sum_scale += out["Dtrajectory_time"]
        nsteps += 1
        t += Tc

    viol_rate = violations / max(1, nsteps)
    mean_scale = sum_scale / max(1, nsteps)
    return viol_rate, mean_scale

# -------------------- OPTUNA OPTIMIZATION --------------------
def objective(trial):
    l1 = trial.suggest_float("lambda1", 1e-2, 1e4, log=True)
    l2 = trial.suggest_float("lambda2", 1e-3, 1e3, log=True)
    l3 = trial.suggest_float("lambda3", 1e-4, 1e2, log=True)
    l4 = trial.suggest_float("lambda4", 1e-14, 1e-2, log=True)
    viol_rate, mean_scale = run_episode(l1, l2, l3, l4)
    return viol_rate, mean_scale  # minimize violations, maximize scaling

study = optuna.create_study(directions=["minimize", "maximize"])
study.optimize(objective, n_trials=40, show_progress_bar=True)

print("Pareto-optimal trials:")
for t in study.best_trials:
    print(f"Trial {t.number} values={t.values} params={t.params}")
