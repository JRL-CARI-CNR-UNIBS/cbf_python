import optuna
import optunahub
import numpy as np
import math
import time
import pinocchio as pin
from joint_interpolator import SegmentedJointTrap
from sharework import loadSharework
from fake_command_bridge import FakeCommandBridge
from optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
from plotly.io import show
from multiprocessing import Process, Queue
from queue import Empty
# Database connection (for dashboard)
POSTGRES_URL = "postgresql+psycopg2://optuna:optuna_pw@localhost:5432/optuna_db"

def make_objective(h_min, h_max):
    def objective(trial):
        lambda_pos = trial.suggest_float("lambda_pos", 5e2, 5e4, log=True)
        lambda_vel = trial.suggest_float("lambda_vel", 1e-3, 1e3, log=True)
        lambda_scaling = trial.suggest_float("lambda_scaling", 10, 1e4, log=True)
        lambda_acc = trial.suggest_float("lambda_acc", 1e-14, 1e-2, log=True)
        gamma = trial.suggest_float("gamma", 1, 10, log=True)
        delta = trial.suggest_float("delta_deg", 0.1, 5, log=True)

        try:
            viol_rate, mean_scale, mean_traj_err, low_scale_rate = run_episode_with_timeout(
                lambda_pos, lambda_vel, lambda_scaling, lambda_acc, gamma, delta, Tc=2e-3, duration=500.0,
                h_min = h_min, h_max = h_max, timeout=600
            )
        except TimeoutError:
            # For directions: [minimize, maximize, minimize, minimize]
            return (1e9, -1e9, 1e9, 1e9)

        return (viol_rate, mean_scale, mean_traj_err, low_scale_rate)
    return objective

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


# Camera and bridge
# Build camera pose from your INITI snippet
quat = pin.Quaternion(0.83, 0.185, 0.513, 0.12)
quat.normalize()
R = quat.toRotationMatrix()

T_wc = pin.SE3(R, np.array([1.04, -0.93, 2.309]))

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
gen_cfg = ControllerConfig(Tc=Tc)
# # Basic planner reused across trials
q = home.copy()
q10 = np.array([ 31.0, -78.0, 115.0, -127.0, 86.0, -32.0])*np.pi/180.0
q20 =  np.array([ 31.0, -83.0, 98.0, -110.0, 86.0, -32.0])*np.pi/180.0
q22 =  np.array([ 40.0, -126.0, 141.0, -100.0, 86.0, 45.0])*np.pi/180.0
q25 =  np.array([ 130.0, -100.0, 125.0, -115.0, 94.0, -20.0])*np.pi/180.0
q30 =  np.array([ 136.0, -60.0, 90.0, -122.0, 90.0, 45.0])*np.pi/180.0
q40 =  np.array([ 134.0, -65.0, 70.0, -90.0, 90.0, 45.0])*np.pi/180.0
gen_cfg.Dq_max = gen_cfg.Dq_max*0.25
gen_cfg.DDq_max = gen_cfg.DDq_max*0.2
planner = SegmentedJointTrap(Dq_max=gen_cfg.Dq_max*0.25, DDq_max=gen_cfg.DDq_max*0.25)
# CONFIG 1
planner.addWayPoint(q)
planner.addWayPoint(q10)
planner.addWayPoint(q20)
planner.addWayPoint(q10)
planner.addWayPoint(q22)
planner.addWayPoint(q25)
planner.addWayPoint(q30)
planner.addWayPoint(q40)
planner.addWayPoint(q30)
planner.addWayPoint(q)


T_total = planner.computeTime()
model_wrapper = loadSharework(UR10E_JOINTS)
model = model_wrapper.model
data = model.createData()
n_wp = 9
configs = {
    "q": q,
    "q10": q10,
    "q20": q20,
    "q22": q22,
    "q25": q25,
    "q30": q30,
    "q40": q40,
}
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
for name in cartesian_configs:
    p, R, T_ee = compute_ee_pose(configs[name], model, data, tool_frame_id)
    cartesian_configs[name] = p.tolist()

scaling_threshold = 0.5  
#-------------------- EVALUATION FUNCTION --------------------
def run_episode(lambda_pos, lambda_vel, lambda_scaling, lambda_acc, gamma, delta, Tc=2e-3, duration=500.0, h_min = -1000.0, h_max = 1000.0):

    home = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0

    cfg = ControllerConfig(Tc=Tc)
    cfg.lambda_pos = lambda_pos
    cfg.lambda_vel = lambda_vel
    cfg.lambda_scaling = lambda_scaling
    cfg.lambda_acc = lambda_acc
    cfg.delta_q_max[0:2] = np.deg2rad(np.array([1,1], dtype=np.float64) * delta)
    cfg.delta_q_max[2:4] = np.deg2rad(np.array([1,1], dtype=np.float64) * delta)*2
    cfg.delta_q_max[4:6] = np.deg2rad(np.array([1,1], dtype=np.float64) * delta)*4
    cfg.Dq_max = cfg.Dq_max*0.25
    cfg.DDq_max = cfg.DDq_max*0.2
    cfg.gamma = gamma

    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True)
   
    # quat = pin.Quaternion(0.814, 0.178, 0.535, 0.137)
    # quat.normalize()
    # R = quat.toRotationMatrix()

    # T_wc = pin.SE3(R, np.array([0.108, -0.883, 2.351]))

    bridge = FakeCommandBridge(
        UR10E_JOINTS,
        csv_path="/home/nyquist/projects/cells_ws/src/zed_skeleton_kinematics/csv_files/skeleton_vectors_14_NORMAL_TEST1.csv",
        Tworld_to_cam=T_wc,
        slowdown_factor=1.0,
        t0=0.0
    )

    ctrl.reset_state(q)
    t = 0.0
    trajectory_time = 0.0
    violations, nsteps = 0, 0
    sum_scale = 0.0
    trajectory_error_sum = 0.0
    lap_count = 0
    on_target_count = 0
    prec_target = -1
    enable_lap_count = True

    low_scale_count = 0
    while t < duration:
        obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed = t)
        nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)
        try:
            out = ctrl.step(
                obs_pos=obs_pos,
                obs_vel=obs_vel,
                obs_acc=obs_acc,
                nominal_q=nominal_q,
                nominal_Dq=nominal_Dq,
                nominal_DDq=nominal_DDq,
            )
            end_eff_pos = out["end_effector_pos"]
            if (trajectory_time % T_total) < Tc:
                if enable_lap_count:
                    lap_count += 1
                    prec_target = -1
                    enable_lap_count = False
            else:
                enable_lap_count = True
            for i in range(len(cartesian_configs.values())):
                q_wp = list(cartesian_configs.values())[i]
                if  np.linalg.norm(q_wp - end_eff_pos) < 2e-03 and prec_target != i:
                    on_target_count += 1
                    prec_target = i
                    break
        except Exception:
            # Penalize infeasible or divergent QP
            print("QP failed")
            return 1.0, -1.0, 1000.0, 1.0
        t += Tc
        
        if h_min <= out["h_min"] <= h_max:
            if out["h_min"] < 0 and out["vr_min"] < -1e-3:
                violations += 1
            sum_scale += out["Dtrajectory_time"]
            nsteps += 1
            trajectory_error_sum += out["trajectory_error"]
            if out["Dtrajectory_time"] < scaling_threshold:
                low_scale_count += 1
        trajectory_time = out["trajectory_time"]
        

       
        time.sleep(1e-4)  # To avoid locking issues in multiprocessing


    #on_target_rate = on_target_count/(n_wp * ((lap_count)+ ((trajectory_time % T_total)/T_total)))
    lap_count = lap_count + ((trajectory_time % T_total)/T_total)
    viol_rate = violations / max(1, nsteps)
    mean_scale = sum_scale / max(1, nsteps)
    mean_trajectory_error = trajectory_error_sum / max(1, nsteps)
    low_scale_rate = low_scale_count / max(1, nsteps)
    return viol_rate, mean_scale, mean_trajectory_error, low_scale_rate


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

h_dic = {
    "h_negative": [-1000.0, 0.0],
    "h_q1": [0.0, 0.25],
    "h_q2": [0.25, 0.5],
    "h_q3": [0.5, 0.75],
    "h_positive": [0.75, 1000.0],
}
for h_key, (h_min, h_max) in h_dic.items():

    study = optuna.create_study(
        directions=["minimize", "maximize","minimize", "minimize"],
        sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
        storage=storage,
        #load_if_exists=True,
        study_name=f"dynamic_params_{h_key}_{time.strftime('%Y%m%d-%H%M%S')}",
    )
    study.set_metric_names(["violation_rate", "mean_scaling", "mean_trajectory_error", "low_scale_rate"])
    study.optimize(make_objective(h_min, h_max), n_trials=2500, show_progress_bar=True, n_jobs=30, gc_after_trial=True)

    # print (run_episode(1e3,1e3,1e3,1e-3,5,1))