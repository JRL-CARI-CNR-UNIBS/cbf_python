import optuna
import optunahub
import numpy as np
import math
import time
import pinocchio as pin
from interpolator import SegmentedSE3Trap
from sharework import loadSharework
from fake_command_bridge import FakeCommandBridge
from PID_cbf_task_controller import UR10CBFController
from plotly.io import show
from multiprocessing import Process, Queue
from queue import Empty
import gc
# Database connection (for dashboard)
POSTGRES_URL = "postgresql+psycopg2://optuna:optuna_pw@localhost:5432/optuna_db"
# ----------------- STATIC INITIALIZATION (only once) -----------------
def compute_ee_pose(q, model, data, ee_frame_id):
    """
    Compute forward kinematics of the end-effector for joint config q.
    Returns (position, rotation_matrix, SE3).
    """
    # Forward kinematics for all joints
    pin.forwardKinematics(model, data, q)
    # Update frame placements
    pin.updateFramePlacements(model, data)

    T_ee = data.oMf[ee_frame_id]  # SE3 from world (o) to frame (f=tool0)
    p = T_ee.translation          # 3D position
    R = T_ee.rotation             # 3x3 rotation matrix
    return p, R, T_ee


# Build camera pose from your INITI snippet
quat = pin.Quaternion(0.83, 0.185, 0.513, 0.12)
quat.normalize()
R = quat.toRotationMatrix()
# 
T_wc = pin.SE3(R, np.array([0.094, -0.93, 2.309]))

home = np.array([90, -140, 140, -90, 90, 0]) * np.pi / 180.0
UR10E_JOINTS = [
    "ur10e_shoulder_pan_joint",
    "ur10e_shoulder_lift_joint",
    "ur10e_elbow_joint",
    "ur10e_wrist_1_joint",
    "ur10e_wrist_2_joint",
    "ur10e_wrist_3_joint",
]



Tc =2e-3
# # Basic planner reused across trials
q = home.copy()
q10 = np.array([ 31.0, -78.0, 115.0, -127.0, 86.0, -32.0])*np.pi/180.0
q20 =  np.array([ 31.0, -83.0, 98.0, -110.0, 86.0, -32.0])*np.pi/180.0
q22 =  np.array([ 40.0, -126.0, 141.0, -100.0, 86.0, 45.0])*np.pi/180.0
q25 =  np.array([ 130.0, -100.0, 125.0, -115.0, 94.0, -20.0])*np.pi/180.0
q30 =  np.array([ 136.0, -60.0, 90.0, -122.0, 90.0, 45.0])*np.pi/180.0
q40 =  np.array([ 134.0, -65.0, 70.0, -90.0, 90.0, 45.0])*np.pi/180.0

v_lin_max = 26.6586 * 0.1 * 0.055  # linear velocity [m/s]
w_max = (44.1351 * 0.1 * 0.055)  # angular velocity [rad/s]

a_lin_max = 650 * 0.1 * 0.1  # linear acceleration [m/s^2]
alpha_max = 750 * 0.1 * 0.1  # angular acceleration [rad/s^2]

planner = SegmentedSE3Trap(vlin_max=v_lin_max*2.4, vang_max=w_max*2.4,
                               alin_max=a_lin_max*1.1, aang_max=alpha_max*1.1)
# CONFIG 1
configs = {
    "q": q,
    "q10": q10,
    "q20": q20,
    "q22": q22,
    "q25": q25,
    "q30": q30,
    "q40": q40,
}
ordered_configs = []

ordered_configs.extend(["q", "q10", "q20", "q10", "q22", "q25", "q30", "q40", "q30", "q"])

model_wrapper = loadSharework(UR10E_JOINTS)
original_model = model_wrapper.model
model = model_wrapper.model.copy()
data = model.createData()
n_wp = 9

cartesian_configs = {
    "q": 0.0,
    "q10": 0.0,
    "q20": 0.0,
    "q22": 0.0,
    "q25": 0.0,
    "q30": 0.0,
    "q40": 0.0,
}
tool_frame_name = "ur10e_wrist_3_joint"
tool_frame_id = model.getFrameId(tool_frame_name)
data = model.createData()

for name in ordered_configs:
    p, R, T_ee = compute_ee_pose(configs[name], model, data, tool_frame_id)
    planner.addWayPoint(T_ee)
T_total = planner.computeTime()
print(f"T_total = {T_total} s")

for name in cartesian_configs:
    p, R, T_ee = compute_ee_pose(configs[name], model, data, tool_frame_id)
    cartesian_configs[name] = p.tolist()

frame_ids = []
for name in [ "ur10e_elbow_joint", "ur10e_wrist_3_joint",]:

    # --- Frame ID (if a frame with that name exists) ---
    try:
        fid = model.getFrameId(name)
    except Exception:
        fid = None  # no frame with exactly that name
    frame_ids.append(fid)


# -------------------- EVALUATION FUNCTION --------------------
def run_episode(wn, xi, gamma, Tc=2e-3, duration=150.0):

    home = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0

    Kp_tra = np.array([1.0, 1.0, 1.0]) * wn ** 2
    Kd_tra = np.array([1.0, 1.0, 1.0]) * 2.0 * xi * wn
    Kp_rot = np.array([1.0, 1.0, 1.0]) * wn ** 2
    Kd_rot = np.array([1.0, 1.0, 1.0]) * 2.0 * xi * wn

    ctrl = UR10CBFController(
        model=model.copy(),
        tool_frame_name=tool_frame_name,
        frames_ids=frame_ids,
        Tc=Tc,
        Kp_tra=Kp_tra,
        Kd_tra=Kd_tra,
        Kp_rot=Kp_rot,
        Kd_rot=Kd_rot,
        gamma=gamma,
    )
    # quat = pin.Quaternion(0.814, 0.178, 0.535, 0.137)
    # quat.normalize()
    # R = quat.toRotationMatrix()

    # T_wc = pin.SE3(R, np.array([0.108, -0.883, 2.351]))

    bridge = FakeCommandBridge(
        UR10E_JOINTS,
        csv_path="/home/nyquist/projects/cells_ws/src/zed_skeleton_kinematics/csv_files/skeleton_vectors.csv",
        Tworld_to_cam=T_wc,
        slowdown_factor=1.0,
        t0 = 0.0,
    )

    try:
        ctrl.reset_state(home)
        t = 0.0
        trajectory_time = 0.0
        Dtrajectory_time = 1.0
        DDtrajectory_time = 0.0
        lap_count = 0
        on_target_count = 0
        prec_target = -1
        violations, nsteps = 0, 0
        sum_scale = 0.0
        trajectory_error_sum = 0.0
        while t < duration:

            if T_total >0.0:
                goal_pose, nominal_twist_goal, nominal_goal_dtwist = planner.getMotionLaw(
                    trajectory_time % T_total
                )
            else:
                goal_pose, nominal_twist_goal, nominal_goal_dtwist = planner.getMotionLaw(
                trajectory_time
                )
            if 0 < (trajectory_time % T_total) < Tc:
                lap_count += 1
                print("LAP ADDED")

            obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles(elapsed=t)
            
            # Scale if you ever implement time-scaling; currently D=1, DD=0
            twist_goal = nominal_twist_goal * Dtrajectory_time
            goal_dtwist = (
                nominal_goal_dtwist * Dtrajectory_time ** 2.0
                + nominal_twist_goal * DDtrajectory_time
            )

            # --------- Controller step (this is the key new API) ------------- #
            out = ctrl.step(
                t=t,
                goal_pose=goal_pose,
                twist_goal=twist_goal,
                goal_dtwist=goal_dtwist,
                obstacle_positions=obstacle_positions,
                obstacle_velocities=obstacle_velocities,
                obstacle_accelerations=obstacle_accelerations,
            )

            q = out["q"]
            dq = out["dq"]
            ddq = out["ddq"]
            h_min = out["h_min"]

            # --------------------------- TIMING & VISUALS ------------------- #
  
            Dtrajectory_time += DDtrajectory_time * Tc

            end_eff_pos = out["end_effector_pos"]


            for i in range(len(cartesian_configs.values())):
                q_wp = list(cartesian_configs.values())[i]
                if np.linalg.norm(q_wp - end_eff_pos) < 2e-03 and prec_target != i:
                    on_target_count += 1
                    prec_target = i
                    # print("TARGET REACHED")
                    break
          
            if out["h_min"] < 0 and out["vr_min"] < -1e-3:
                violations += 1
            # print(f"violations = {violations}, nsteps = {nsteps}, h_min = {out['h_min']}")
            # sum_scale += Dtrajectory_time
            nsteps += 1
            t += Tc
            trajectory_time = t
            trajectory_error_sum += out["trajectory_error"]
            time.sleep(1e-4)  # To avoid locking issues in multiprocessing

    except Exception as e:
            # Penalize infeasible or divergent QP
            print("QP failed, exception:", e)
            return 1.0, 1000.0, -1.0
    print ("FINE CICLO")        
    on_target_rate = on_target_count/(n_wp * ((lap_count)+ ((trajectory_time % T_total)/T_total)))
    # lap_count = lap_count + ((trajectory_time % T_total)/T_total)
    viol_rate = violations / max(1, nsteps)
    mean_trajectory_error = trajectory_error_sum / max(1, nsteps)
    ctrl.close()
    del ctrl
    gc.collect()
    return viol_rate,  mean_trajectory_error, on_target_rate


def _run_episode_worker(args, kwargs, q):
    """Runs run_episode and returns either ('ok', result) or ('err', repr(exception))."""
    try:
        result = run_episode(*args, **kwargs)
        q.put(("ok", result))
    except Exception as e:
        q.put(("err", repr(e)))

def run_episode_with_timeout(*args, timeout=600, **kwargs):
    """
    Run run_episode(...), but stop it if it takes longer than `timeout` seconds.
    Returns the tuple from run_episode on success.
    Raises TimeoutError if exceeded, or RuntimeError if the worker failed.
    """
    q = Queue()
    p = Process(target=_run_episode_worker, args=(args, kwargs, q), daemon=True)
    p.start()
    p.join(timeout)

    if p.is_alive():
        # Hard timeout: terminate the process and clean up
        p.terminate()
        p.join()
        raise TimeoutError(f"run_episode exceeded {timeout}s and was terminated")

    try:
        status, payload = q.get_nowait()
    except Empty:
        raise RuntimeError("Worker exited without returning a result (crash or early exit).")

    if status == "ok":
        return payload
    else:
        raise RuntimeError(f"run_episode failed in worker: {payload}")

# -------------------- OPTUNA OPTIMIZATION --------------------
def objective(trial):
    wn = trial.suggest_float("wn", low=10, high=200, log=True)
    xi = trial.suggest_float("xi", low=0.01, high=10, log=True)
    gamma = trial.suggest_float("gamma", low=1, high=10, log=True)
    try:
        viol_rate, mean_scale, mean_trajectory_error, lap_count, on_target_rate = run_episode_with_timeout(wn, xi, gamma)
    except TimeoutError as e:
        print(f"Trial timed out: {e}")
        return 1.0, 1000.0, 0.0  # Penalize timeout

    return viol_rate, mean_trajectory_error, on_target_rate


storage = optuna.storages.RDBStorage(
    url=POSTGRES_URL,
    engine_kwargs={
        "pool_pre_ping": True,
        "pool_size": 40,
        "max_overflow": 20,
    },
    heartbeat_interval=30,     # worker pings DB every 30s
    grace_period=120,          # declare failed if no ping for 120s
    # failed_trial_callback=RetryFailedTrialCallback(max_retry=1),
)

study = optuna.create_study(
    directions=["minimize","minimize", "maximize"],
    sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
    storage=storage,
    load_if_exists=True,
    study_name=f"dynamic_params_PID_study_{time.strftime('%Y%m%d-%H%M%S')}",
)
study.optimize(objective, n_trials=2500, show_progress_bar=True, n_jobs=30)

# print(run_episode(40, 0.33, 5))  # For debugging