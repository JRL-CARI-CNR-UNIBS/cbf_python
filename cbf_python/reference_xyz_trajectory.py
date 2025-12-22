import pandas as pd
import numpy as np
import pinocchio as pin
from sharework import loadSharework

# Load the model
UR10E_JOINTS = [
    "ur10e_shoulder_pan_joint",
    "ur10e_shoulder_lift_joint",
    "ur10e_elbow_joint",
    "ur10e_wrist_1_joint",
    "ur10e_wrist_2_joint",
    "ur10e_wrist_3_joint",
]
model_wrapper = loadSharework(UR10E_JOINTS)
model = model_wrapper.model
data = model.createData()
# Load joint states CSV
joint_state_df = pd.read_csv("resullts/simulation/scaling/20251219_155319/reference_trajectory_2025_12_19_15_53_19.csv", header=0, index_col=False)


# Function to compute the Cartesian state for the wrist joint
def get_wrist_cartesian_state(joint_angles):
    # Create the configuration vector for the joint angles
    q = np.array(joint_angles)  # Assuming joint_angles is a list or np.array of joint positions
    #print(q)
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    # Get the transformation matrix for the wrist 3 joint (ur10e_wrist_3_joint)
    frame_id = model.getFrameId('ur10e_tool0')
    transformation_matrix = data.oMf[frame_id]
    # Extract the position (translation part) and orientation (rotation part)
    position = transformation_matrix.translation
    orientation = transformation_matrix.rotation
    # Convert rotation matrix to quaternion
    quaternion = pin.Quaternion(orientation)
    return position, quaternion


# Prepare the data to save
cartesian_data = []

# Iterate over each row in the joint state log and compute the wrist Cartesian state
for index, row in joint_state_df.iterrows():
    # Extract joint positions from the CSV (assuming joint positions are at columns like 'joint_0_pos', 'joint_1_pos', etc.)
    joint_positions = [row[f"target_joint_{i}_pos"] for i in range(len(UR10E_JOINTS))]

    # Get the wrist 3 Cartesian position and orientation
    position, orientation = get_wrist_cartesian_state(joint_positions)
    # Append the result to the list
    cartesian_data.append([row['time'], *position, orientation.x, orientation.y, orientation.z, orientation.w])
print (cartesian_data[0])
# Convert the list to a DataFrame
cartesian_df = pd.DataFrame(cartesian_data, columns=['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

# Save the Cartesian state to a new CSV
cartesian_df.to_csv("resullts/simulation/scaling/20251219_155319/ur10e_wrist_3_cartesian_state.csv", index=False)

print("Cartesian data for wrist 3 joint saved to 'ur10e_wrist_3_cartesian_state.csv'")
