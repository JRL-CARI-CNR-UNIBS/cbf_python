#!/usr/bin/env python3

import functools

import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_receive
import rtde_control
import math
import signal


from concurrent.futures import ThreadPoolExecutor

from robotiq_socket import RobotiqCModelURCap
from openpi_client import image_tools, websocket_client_policy

from cbf_python.scripts.example_cbf_optimal import load_config, setup_controller, _handle_sigint
from sharework import loadSharework
from cbf_python.Command_bridge.joint_command_bridge import JointStateCommandBridge
# ============================================================
# CONFIG
# ============================================================

ROBOT_IP = "192.168.10.2"

FRONT_SERIAL = "116622071830"
WRIST_SERIAL = "922612070587"

SERVER_HOST = "localhost"
SERVER_PORT = 8000

CAM_W = 640
CAM_H = 480
CAM_FPS = 30
IMG_SIZE = 224

PROMPT = "Pick up the black metal object and place it in the black box"
SAFETY_PROMPT = "Stay away from the human"

# ============================================================
# EXECUTION
# ============================================================

# UR servo loop
SERVO_HZ = 200
SERVO_DT = 1.0 / SERVO_HZ

# Policy action execution
ACTION_HZ = 20

# 200 / 10 = exactly 20 servo commands per policy action
SERVO_STEPS_PER_ACTION = SERVO_HZ // ACTION_HZ

# π0.5 currently outputs 30 actions
ACTION_HORIZON = 30

# Use the complete prediction unless early replanning is needed
EXECUTE_ACTIONS = 30

LOOKAHEAD_TIME = 0.10
GAIN = 100

# First action after every new inference gets extra time
FIRST_TARGET_TIME = 0.50


# ============================================================
# NON-FATAL POLICY LIMITS
#
# These limits CLAMP the trajectory.
# They do not terminate execution.
# ============================================================

MAX_FIRST_DELTA = 0.25

MAX_STEP_DELTA = 0.05

MAX_CHUNK_DELTA = 0.45


# ============================================================
# EARLY REPLAN
#
# This also does NOT terminate the program.
#
# If the robot falls behind by more than this:
#
#     abandon remaining actions
#     -> observe
#     -> infer again
# ============================================================

REPLAN_TRACKING_ERROR = 0.15


# ============================================================
# GRIPPER
# ============================================================

# Dataset convention:
#
# 0 = open
# 1 = closed

GRIPPER_SPEED = 40
GRIPPER_FORCE = 150

GRIPPER_MIN_CHANGE = 2


# ============================================================
# CAMERA
# ============================================================

def start_camera(serial):

    pipe = rs.pipeline()
    config = rs.config()

    config.enable_device(serial)

    config.enable_stream(
        rs.stream.color,
        CAM_W,
        CAM_H,
        rs.format.bgr8,
        CAM_FPS,
    )

    pipe.start(config)

    # Warm up camera
    for _ in range(10):
        pipe.wait_for_frames()

    return pipe


def get_image(pipe):

    frames = pipe.wait_for_frames()

    color = frames.get_color_frame()

    if not color:
        raise RuntimeError(
            "No camera frame received"
        )

    bgr = np.asanyarray(
        color.get_data()
    )

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    rgb = image_tools.resize_with_pad(
        rgb,
        IMG_SIZE,
        IMG_SIZE,
    )

    return image_tools.convert_to_uint8(
        rgb
    )


# ============================================================
# GRIPPER
# ============================================================

def norm_to_gripper_raw(value):

    value = float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )

    return int(
        round(
            value * 255.0
        )
    )


def command_gripper(
    gripper,
    normalized_target,
    previous_raw,
):

    raw_target = norm_to_gripper_raw(
        normalized_target
    )

    if (
        previous_raw is None
        or abs(
            raw_target - previous_raw
        ) >= GRIPPER_MIN_CHANGE
    ):

        gripper.move(
            raw_target,
            GRIPPER_SPEED,
            GRIPPER_FORCE,
        )

        print(
            f" | gripper={normalized_target:.3f}"
            f" raw={raw_target}",
            end="",
        )

        return raw_target

    return previous_raw


# ============================================================
# OBSERVATION + INFERENCE
# ============================================================

def capture_and_infer(
    policy,
    rtde_r,
    gripper,
    front_cam,
    wrist_cam,
):

    # --------------------------------------------------------
    # Robot state
    # --------------------------------------------------------

    q_now = np.asarray(
        rtde_r.getActualQ(),
        dtype=np.float32,
    )

    gripper_raw = (
        gripper.get_current_position()
    )

    gripper_norm = (
        float(gripper_raw)
        / 255.0
    )

    state = np.concatenate(
        [
            q_now,
            [gripper_norm],
        ]
    ).astype(np.float32)


    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    front_rgb = get_image(
        front_cam
    )

    wrist_rgb = get_image(
        wrist_cam
    )


    # --------------------------------------------------------
    # Policy input
    # --------------------------------------------------------

    observation = {
        "state": state,
        "base_rgb": front_rgb,
        "wrist_rgb": wrist_rgb,
        "prompt": PROMPT,
    }


    # --------------------------------------------------------
    # π0.5
    # --------------------------------------------------------

    result = policy.infer(
        observation
    )

    actions = np.asarray(
        result["actions"],
        dtype=np.float64,
    )

    return actions, q_now


# ============================================================
# PREPARE / CLAMP POLICY TRAJECTORY
# ============================================================

def prepare_policy_chunk(
    actions,
    q_now,
):

    q_now = np.asarray(
        q_now,
        dtype=np.float64,
    )


    # ========================================================
    # Hard validity checks
    #
    # Invalid policy output should still terminate because
    # continuing from corrupt data is not useful.
    # ========================================================

    if actions.shape != (
        ACTION_HORIZON,
        7,
    ):

        raise RuntimeError(
            f"Expected "
            f"({ACTION_HORIZON}, 7), "
            f"got {actions.shape}"
        )


    if not np.all(
        np.isfinite(actions)
    ):

        raise RuntimeError(
            "Policy returned NaN or Inf"
        )


    # ========================================================
    # Select chunk
    # ========================================================

    raw_chunk = actions[
        :EXECUTE_ACTIONS
    ].copy()

    raw_joints = (
        raw_chunk[:, :6]
    )

    gripper_actions = np.clip(
        raw_chunk[:, 6],
        0.0,
        1.0,
    )


    # ========================================================
    # Diagnostics BEFORE clamping
    # ========================================================

    raw_first_delta = float(
        np.max(
            np.abs(
                raw_joints[0]
                - q_now
            )
        )
    )


    if EXECUTE_ACTIONS > 1:

        raw_step_delta = float(
            np.max(
                np.abs(
                    np.diff(
                        raw_joints,
                        axis=0,
                    )
                )
            )
        )

    else:

        raw_step_delta = 0.0


    raw_chunk_delta = float(
        np.max(
            np.abs(
                raw_joints
                - q_now
            )
        )
    )


    full_horizon_delta = float(
        np.max(
            np.abs(
                actions[:, :6]
                - q_now
            )
        )
    )


    # ========================================================
    # SAFE TRAJECTORY
    # ========================================================

    safe_joints = np.zeros_like(
        raw_joints
    )


    # --------------------------------------------------------
    # ACTION 0
    #
    # Limit first policy target relative to current actual
    # robot position.
    # --------------------------------------------------------

    safe_joints[0] = np.clip(
        raw_joints[0],
        q_now - MAX_FIRST_DELTA,
        q_now + MAX_FIRST_DELTA,
    )


    # --------------------------------------------------------
    # ACTIONS 1 ... N
    # --------------------------------------------------------

    for i in range(
        1,
        EXECUTE_ACTIONS,
    ):

        # Limit total displacement from observed position
        absolute_limited = np.clip(
            raw_joints[i],
            q_now - MAX_CHUNK_DELTA,
            q_now + MAX_CHUNK_DELTA,
        )


        # Limit action-to-action jump
        safe_joints[i] = np.clip(
            absolute_limited,
            safe_joints[i - 1]
            - MAX_STEP_DELTA,
            safe_joints[i - 1]
            + MAX_STEP_DELTA,
        )


        # Guarantee total chunk boundary
        safe_joints[i] = np.clip(
            safe_joints[i],
            q_now - MAX_CHUNK_DELTA,
            q_now + MAX_CHUNK_DELTA,
        )


    # ========================================================
    # Diagnostics AFTER clamping
    # ========================================================

    safe_first_delta = float(
        np.max(
            np.abs(
                safe_joints[0]
                - q_now
            )
        )
    )


    if EXECUTE_ACTIONS > 1:

        safe_step_delta = float(
            np.max(
                np.abs(
                    np.diff(
                        safe_joints,
                        axis=0,
                    )
                )
            )
        )

    else:

        safe_step_delta = 0.0


    safe_chunk_delta = float(
        np.max(
            np.abs(
                safe_joints
                - q_now
            )
        )
    )


    changed = np.any(
        np.abs(
            safe_joints
            - raw_joints
        ) > 1e-8,
        axis=1,
    )

    number_clamped = int(
        np.sum(changed)
    )


    # ========================================================
    # Print
    # ========================================================

    print(
        f"Raw first delta      : "
        f"{raw_first_delta:.5f} rad"
    )

    print(
        f"Raw step delta       : "
        f"{raw_step_delta:.5f} rad"
    )

    print(
        f"Raw {EXECUTE_ACTIONS} delta"
        f"      : "
        f"{raw_chunk_delta:.5f} rad"
    )

    print(
        f"Full horizon delta   : "
        f"{full_horizon_delta:.5f} rad"
    )


    if number_clamped:

        print(
            f"CLAMPED              : "
            f"{number_clamped}/"
            f"{EXECUTE_ACTIONS} actions"
        )

        print(
            f"Safe first delta     : "
            f"{safe_first_delta:.5f} rad"
        )

        print(
            f"Safe step delta      : "
            f"{safe_step_delta:.5f} rad"
        )

        print(
            f"Safe chunk delta     : "
            f"{safe_chunk_delta:.5f} rad"
        )

    else:

        print(
            "CLAMPED              : none"
        )


    print(
        f"Gripper next "
        f"{EXECUTE_ACTIONS}"
        f"      : "
        f"{np.min(gripper_actions):.3f}"
        f" -> "
        f"{np.max(gripper_actions):.3f}"
    )


    return (
        safe_joints,
        gripper_actions,
    )


# ============================================================
# SERVO TARGET
# ============================================================

def servo_to_target(
    rtde_c,
    rtde_r,
    q_start,
    q_target,
    steps,
):

    q_start = np.asarray(
        q_start,
        dtype=np.float64,
    )

    q_target = np.asarray(
        q_target,
        dtype=np.float64,
    )


    # ========================================================
    # 150 Hz interpolation
    # ========================================================

    for step in range(
        1,
        steps + 1,
    ):

        alpha = (
            step / steps
        )

        q_cmd = (
            q_start
            + alpha
            * (
                q_target
                - q_start
            )
        )


        period = (
            rtde_c.initPeriod()
        )


        rtde_c.servoJ(
            q_cmd.tolist(),
            0.0,
            0.0,
            SERVO_DT,
            LOOKAHEAD_TIME,
            GAIN,
        )


        rtde_c.waitPeriod(
            period
        )


    # ========================================================
    # Measure what actually happened
    # ========================================================

    q_actual = np.asarray(
        rtde_r.getActualQ(),
        dtype=np.float64,
    )


    error_per_joint = np.abs(
        q_target
        - q_actual
    )


    tracking_error = float(
        np.max(
            error_per_joint
        )
    )


    worst_joint = int(
        np.argmax(
            error_per_joint
        )
    )


    # IMPORTANT:
    #
    # No RuntimeError here.
    #
    # The caller decides whether this should trigger a new
    # π0.5 inference.
    # ========================================================

    return (
        q_actual,
        tracking_error,
        worst_joint,
    )


# ============================================================
# HOLD WHILE π0.5 IS RUNNING
# ============================================================

def hold_while_inferencing(
    rtde_c,
    future,
    hold_target,
):

    hold_target = np.asarray(
        hold_target,
        dtype=np.float64,
    )


    while not future.done():

        period = (
            rtde_c.initPeriod()
        )


        rtde_c.servoJ(
            hold_target.tolist(),
            0.0,
            0.0,
            SERVO_DT,
            LOOKAHEAD_TIME,
            GAIN,
        )


        rtde_c.waitPeriod(
            period
        )


# ============================================================
# MAIN
# ============================================================

def main():

    rtde_r = None
    rtde_c = None

    gripper = None

    front_cam = None
    wrist_cam = None

    executor = None


    try:

        # ====================================================
        # CONNECTIONS
        # ====================================================

        print(
            "Connecting to UR10e..."
        )

        rtde_r = (
            rtde_receive.RTDEReceiveInterface(
                ROBOT_IP,
                frequency=SERVO_HZ,
            )
        )


        print(
            "Connecting to gripper..."
        )

        gripper = (
            RobotiqCModelURCap(
                ROBOT_IP
            )
        )


        print(
            "Starting front camera..."
        )

        front_cam = start_camera(
            FRONT_SERIAL
        )


        print(
            "Starting wrist camera..."
        )

        wrist_cam = start_camera(
            WRIST_SERIAL
        )


        print(
            "Connecting to policy server..."
        )

        policy = (
            websocket_client_policy
            .WebsocketClientPolicy(
                host=SERVER_HOST,
                port=SERVER_PORT,
            )
        )


        executor = ThreadPoolExecutor(
            max_workers=1
        )


        # ====================================================
        # INITIAL OBSERVATION + INFERENCE
        # ====================================================

        print(
            "\nRunning initial policy inference..."
        )


        actions, q_observed = (
            capture_and_infer(
                policy,
                rtde_r,
                gripper,
                front_cam,
                wrist_cam,
            )
        )


        joint_actions, gripper_actions = (
            prepare_policy_chunk(
                actions,
                q_observed,
            )
        )


        # ====================================================
        # INFO
        # ====================================================

        print(
            "\n" + "=" * 75
        )

        print(
            "π0.5 CLOSED-LOOP EXECUTION"
        )

        print(
            "=" * 75
        )

        print(
            f"Servo rate          : "
            f"{SERVO_HZ} Hz"
        )

        print(
            f"Action rate         : "
            f"{ACTION_HZ} Hz"
        )

        print(
            f"Servo steps/action  : "
            f"{SERVO_STEPS_PER_ACTION}"
        )

        print(
            f"Policy horizon      : "
            f"{ACTION_HORIZON}"
        )

        print(
            f"Max actions/infer   : "
            f"{EXECUTE_ACTIONS}"
        )

        print(
            f"Max chunk duration  : "
            f"{EXECUTE_ACTIONS / ACTION_HZ:.2f} s"
        )

        print(
            f"Replan error        : "
            f"{REPLAN_TRACKING_ERROR:.3f} rad"
        )

        print(
            "Tracking error action: "
            "REPLAN, NOT STOP"
        )

        print(
            "Automatic cycle stop: NONE"
        )


        input(
            "\nKeep emergency stop ready.\n"
            "Press ENTER to begin..."
        )


        # ====================================================
        # RTDE CONTROL
        # ====================================================

        print(
            "\nConnecting RTDE control..."
        )

        rtde_c = (
            rtde_control.RTDEControlInterface(
                ROBOT_IP,
                SERVO_HZ,
            )
        )


        last_gripper_raw = (
            gripper.get_current_position()
        )

        # ====================================================
        # CBF CONTROLLER INITIALIZATION
        # ====================================================
        
        config = load_config()
        robot_cfg = config.get("robot", {})
        joint_names = robot_cfg.get("joint_names", [
                "ur10e_shoulder_pan_joint",
                "ur10e_shoulder_lift_joint",
                "ur10e_elbow_joint",
                "ur10e_wrist_1_joint",
                "ur10e_wrist_2_joint",
                "ur10e_wrist_3_joint",
            ])
        
        # 1. Load Robot Kinematic & Dynamic Model
        model_wrapper = loadSharework(joint_names)
        model = model_wrapper.model
    
        # 2. Configure Controller
        cfg, ctrl = setup_controller(model_wrapper, config)
        print(cfg)

        bridge_cfg = config.get("bridge", {})
        threshold = float(bridge_cfg.get("threshold", 1.1))
        timeout_sec = float(bridge_cfg.get("timeout_sec", 5.0))
        bridge = JointStateCommandBridge(
            ordered_joint_names=joint_names,
            threshold=threshold,
        )
        tool_frame_name = cfg.tool_frame

        first_joint_position = bridge.wait_for_first_state(tool_frame_name, timeout=timeout_sec)
        signal.signal(signal.SIGINT, functools.partial(_handle_sigint, bridge))
        if math.isnan(first_joint_position):
            bridge.shutdown()
            return
        first_joint_position = bridge.getPositions()
        # bridge.switch_to_forward_position_controller_service()

        # ====================================================
        # CONTINUOUS CLOSED LOOP
        # ====================================================

        cycle = 1


        while True:

            print(
                "\n" + "=" * 75
            )

            print(
                f"CYCLE {cycle}"
            )

            print(
                "=" * 75
            )


            # ------------------------------------------------
            # Current actual robot state
            # ------------------------------------------------

            q_actual = np.asarray(
                rtde_r.getActualQ(),
                dtype=np.float64,
            )


            early_replan = False

            actions_executed = 0


            # =================================================
            # EXECUTE POLICY TRAJECTORY
            # =================================================

            for action_index in range(
                EXECUTE_ACTIONS
            ):

                q_target = (
                    joint_actions[
                        action_index
                    ]
                )


                # --------------------------------------------
                # Gripper
                # --------------------------------------------

                last_gripper_raw = (
                    command_gripper(
                        gripper,
                        gripper_actions[
                            action_index
                        ],
                        last_gripper_raw,
                    )
                )


                # --------------------------------------------
                # Start from actual measured robot state
                # --------------------------------------------

                q_start = np.asarray(
                    rtde_r.getActualQ(),
                    dtype=np.float64,
                )


                # --------------------------------------------
                # First target gets additional transition time
                # --------------------------------------------

                if action_index == 0:

                    steps = max(
                        1,
                        int(
                            FIRST_TARGET_TIME
                            * SERVO_HZ
                        ),
                    )

                else:

                    steps = (
                        SERVO_STEPS_PER_ACTION
                    )


                # --------------------------------------------
                # Execute
                # --------------------------------------------

                (
                    q_actual,
                    tracking_error,
                    worst_joint,
                ) = servo_to_target(
                    rtde_c,
                    rtde_r,
                    q_start,
                    q_target,
                    steps,
                )


                actions_executed += 1


                print(
                    f"\rAction "
                    f"{action_index:02d}/"
                    f"{EXECUTE_ACTIONS - 1:02d}"
                    f" | error="
                    f"{tracking_error:.5f} rad"
                    f" | J{worst_joint + 1}",
                    end="",
                    flush=True,
                )


                # =================================================
                # IMPORTANT:
                #
                # Large tracking error does NOT stop execution.
                #
                # It abandons the stale remainder of this chunk
                # and immediately asks π0.5 what to do next.
                # =================================================

                if (
                    tracking_error
                    > REPLAN_TRACKING_ERROR
                ):

                    print()

                    print(
                        "\nTracking error exceeded "
                        "replan threshold:"
                    )

                    print(
                        f"  error       = "
                        f"{tracking_error:.5f} rad"
                    )

                    print(
                        f"  worst joint = "
                        f"J{worst_joint + 1}"
                    )

                    print(
                        f"  action      = "
                        f"{action_index}/"
                        f"{EXECUTE_ACTIONS - 1}"
                    )

                    print(
                        "Abandoning remaining "
                        "trajectory."
                    )

                    print(
                        "Requesting fresh π0.5 "
                        "prediction..."
                    )


                    early_replan = True

                    break


            print()


            # =================================================
            # POSITION TO HOLD DURING INFERENCE
            # =================================================

            if early_replan:

                # ------------------------------------------------
                # Robot fell behind.
                #
                # Hold where it ACTUALLY is.
                #
                # Do not continue chasing the previous policy
                # target while a new trajectory is being created.
                # ------------------------------------------------

                hold_target = np.asarray(
                    rtde_r.getActualQ(),
                    dtype=np.float64,
                )

            else:

                # ------------------------------------------------
                # Completed entire chunk normally.
                #
                # Hold actual final position.
                # ------------------------------------------------

                hold_target = np.asarray(
                    rtde_r.getActualQ(),
                    dtype=np.float64,
                )


            # =================================================
            # NEW OBSERVATION + INFERENCE
            # =================================================

            print(
                "Observing scene and "
                "running next inference..."
            )


            future = executor.submit(
                capture_and_infer,
                policy,
                rtde_r,
                gripper,
                front_cam,
                wrist_cam,
            )


            # ------------------------------------------------
            # Keep servoJ alive at 150 Hz during inference
            # ------------------------------------------------

            hold_while_inferencing(
                rtde_c,
                future,
                hold_target,
            )


            # ------------------------------------------------
            # Receive fresh trajectory
            # ------------------------------------------------

            actions, q_observed = (
                future.result()
            )


            joint_actions, gripper_actions = (
                prepare_policy_chunk(
                    actions,
                    q_observed,
                )
            )


            print(
                f"Previous chunk executed: "
                f"{actions_executed}/"
                f"{EXECUTE_ACTIONS}"
            )


            if early_replan:

                print(
                    "Reason: tracking error "
                    "triggered early replanning."
                )

            else:

                print(
                    "Reason: full trajectory "
                    "completed."
                )


            cycle += 1


    # ========================================================
    # USER STOP
    # ========================================================

    except KeyboardInterrupt:

        print(
            "\n\nExecution stopped by user."
        )


    # ========================================================
    # ACTUAL HARD ERROR
    #
    # Examples:
    # camera failure
    # RTDE failure
    # server failure
    # NaN policy output
    # ========================================================

    except Exception as exc:

        print(
            f"\nHARD ERROR: {exc}"
        )


    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        print(
            "\nStopping robot..."
        )


        if rtde_c is not None:

            try:
                rtde_c.servoStop()
            except Exception:
                pass

            try:
                rtde_c.stopScript()
            except Exception:
                pass


        if executor is not None:

            try:
                executor.shutdown(
                    wait=False,
                    cancel_futures=True,
                )
            except Exception:
                pass


        if front_cam is not None:

            try:
                front_cam.stop()
            except Exception:
                pass


        if wrist_cam is not None:

            try:
                wrist_cam.stop()
            except Exception:
                pass


        if rtde_r is not None:

            try:
                rtde_r.disconnect()
            except Exception:
                pass


        if gripper is not None:

            try:
                gripper.disconnect()
            except Exception:
                pass


        print(
            "Done."
        )


if __name__ == "__main__":
    main()