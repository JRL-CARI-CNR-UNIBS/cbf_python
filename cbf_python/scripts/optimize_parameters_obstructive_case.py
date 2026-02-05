import optuna
import optunahub
import numpy as np
import time
import pinocchio as pin
from scripts.util.joint_interpolator import SegmentedJointTrap
from sharework import loadSharework
from Command_bridge.fake_command_bridge import FakeCommandBridge
from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
from multiprocessing import Process, Queue
from queue import Empty

from scripts.util.test_utils import compute_ee_pose, generate_velocity

# Database connection (for dashboard)
POSTGRES_URL = "postgresql+psycopg2://optuna:optuna_pw@localhost:5432/optuna_db"

def make_objective():
    def objective(trial):
        lambda_pos = trial.suggest_float("lambda_pos", 1, 1e6, log=True)
        lambda_vel = trial.suggest_float("lambda_vel", 1e-3, 1e3, log=True)
        lambda_scaling = trial.suggest_float("lambda_scaling", 1e-02, 1e4, log=True)
        lambda_acc = trial.suggest_float("lambda_acc", 1e-14, 1, log=True)
        gamma = trial.suggest_float("gamma", 1, 20, log=True)
        delta = trial.suggest_float("delta_deg", 0.1, 50, log=True)

        try:
            viol_rate, mean_scale, mean_traj_err, low_scale_rate, lap_count = run_episode_with_timeout(
                lambda_pos, lambda_vel, lambda_scaling, lambda_acc, gamma, delta, Tc=2e-3, duration=1000.0,
                timeout=600
            )
        except TimeoutError:
            # For directions: [minimize, maximize, minimize, minimize]
            return (1e9, -1e9, 1e9, 1e9,0.0)

        return (viol_rate, mean_scale, mean_traj_err, low_scale_rate, lap_count)
    return objective

# ----------------- STATIC INITIALIZATION (only once) -----------------
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
planner.addWayPoint(q22)
planner.addWayPoint(q25)
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
def run_episode(lambda_pos, lambda_vel, lambda_scaling, lambda_acc, gamma, delta, Tc=2e-3, duration=500.0):

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
    obstacle_positions = np.zeros(3)
    obstacle_velocities = np.zeros(3)
    obstacle_accelerations = np.zeros(3)
    end_eff_pos = np.zeros(3)
    count_move = 0
    Dtrajectory_time = 1.0

    while t < duration:

        if (nsteps % 500 == 0):
            q_temp, dq_temp, ddq_temp = planner.getMotionLaw((trajectory_time + 2) % T_total)
            obstacle_positions, a, b = compute_ee_pose(q_temp, model, data, tool_frame_id)
            obstacle_positions = obstacle_positions.tolist()
            obstacle_positions[0] = obstacle_positions[0] + 0.0
            obstacle_positions[1] = obstacle_positions[1] + 0.0
            obstacle_positions[2] = obstacle_positions[2] - 0.2
            obstacle_positions = np.array(obstacle_positions)
            obstacle_positions = obstacle_positions.reshape(1, 3)
            obstacle_velocities = generate_velocity(end_eff_pos, obstacle_positions, 0.05)
            obstacle_velocities = obstacle_velocities.reshape(1, 3)
            obstacle_accelerations = obstacle_accelerations.reshape(1, 3)
            count_move = 0
        if Dtrajectory_time < 0.05 and count_move < 40:
            obstacle_positions[0][0] += 0.01
            obstacle_positions[0][1] += 0.01
            count_move += 1
        nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)
        try:
            out = ctrl.step(
                obs_pos=obstacle_positions,
                obs_vel=obstacle_velocities,
                obs_acc=obstacle_accelerations,
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
        
        if out["h_min"] < 0 and out["vr_min"] < -1e-3:
            violations += 1
        sum_scale += out["Dtrajectory_time"]
        nsteps += 1
        trajectory_error_sum += out["trajectory_error"]
        if out["Dtrajectory_time"] < scaling_threshold:
            low_scale_count += 1
        trajectory_time = out["trajectory_time"]

        time.sleep(1e-4)  # To avoid locking issues in multiprocessing


    # on_target_rate = on_target_count/(n_wp * ((lap_count)+ ((trajectory_time % T_total)/T_total)))
    lap_count = lap_count + ((trajectory_time % T_total)/T_total)
    viol_rate = violations / max(1, nsteps)
    mean_scale = sum_scale / max(1, nsteps)
    mean_trajectory_error = trajectory_error_sum / max(1, nsteps)
    low_scale_rate = low_scale_count / max(1, nsteps)
    return viol_rate, mean_scale, mean_trajectory_error, low_scale_rate, lap_count


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


study = optuna.create_study(
    directions=["minimize", "maximize","minimize", "minimize", "maximize"],
    sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
    storage=storage,
    #load_if_exists=True,
    study_name=f"dynamic_params_obstructed case_{time.strftime('%Y%m%d-%H%M%S')}",
)
study.set_metric_names(["violation_rate", "mean_scaling", "mean_trajectory_error", "low_scale_rate", "lap count"])
study.optimize(make_objective(), n_trials=5000, show_progress_bar=True, n_jobs=30, gc_after_trial=True)

# print (run_episode(1e3,1e3,1e3,1e-3,5,1))