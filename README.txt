Per accendere il robot:
premi il tasto accensione
premi turn on + start
metti in modalità remote

In un terminale (ctrl+t)

per lanciare il programm sul robot reale: eseguire il comando 
ros2 launch sharework_cembre_bringup sharework_cembre_bringup.launch.py fake_ur:=false fake_gripper:=true robotiq_use_socket_communication:=false enable_zed_camera:=true enable_dual_realsense:=true launch_rviz:=true

per lanciare il programm sul robot virtuale: eseguire il comando 
ros2 launch sharework_cembre_bringup sharework_cembre_bringup.launch.py fake_ur:=true fake_gripper:=true robotiq_use_socket_communication:=false enable_zed_camera:=true enable_dual_realsense:=true launch_rviz:=true


---------------------
In un terminale (ctrl+t)

ros2 launch zed_skeleton_kinematics zed_skeleton_kinematics.launch.py


---------------------
cd /home/nyquist/projects/tesisti/agnelli
source agnelli/bin/activate

