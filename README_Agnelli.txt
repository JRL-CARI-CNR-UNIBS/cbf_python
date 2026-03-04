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
per scheletonizzazione:
ros2 launch zed_skeleton_kinematics zed_skeleton_kinematics.launch.py


(*per scheletonizzazione e registrazione:
ros2 launch zed_skeleton_kinematics zed_skeleton_kinematics_logging.launch.py fcutoff_arg:=35.0 output_csv_arg:='/home/nyquist/projects/tesisti/agnelli/cbf_python/skeletons_csv' enable_skeleton_logging_arg:=true
*)

---------------------
cd /home/nyquist/projects/tesisti/agnelli
source agnelli/bin/activate


-------------------
In Rviz bisogna fornire il topic: 
add ->  MarkerArray -> topic: /zed/zed_node/body_trk/skeletons_kinematics/vel_markers 
