# Single Object Tracking in ROS

### Code Structure
```
.
├── 03_SOT_In_ROS
│   ├── ros_single_object_tracking.py
```

### Dependencies
#### Python package
1. numpy
2. cv2

#### ROS message and package
1. sensor_msgs
2. vision_msgs
3. geometry_msgs
4. cv_bridge

### Testing
0. Deactivate previous virtual environment if it hsa been activated. 
1. Open 7 terminals and source the ros environment and navigate to bonus task directory. 
    ```bash
    $ . /opt/ros/noetic/setup.bash
    $ cd <HW1/03_SOT_In_ROS>
    ```
2. In first terminal, run `roscore` to start the ros master. 
3. In second terminal, run `rosrun rviz rviz -d ./rviz_cfg.rviz` to launch the rviz with Image view. 
4. In third terminal, run `python3 ros_single_object_tracking.py -s 2` to start the ros node for object tracking on sequence 2. 
5. In fourth, fifth and sixth terminals, run the following commands respectively. 

    `rostopic echo /me5413/groundtruth`

    `rostopic echo /me5413/track`

    `rostopic echo /me5413/nusnetID`
6. In seventh terminal, run `rosbag play ./rosbags/seq2.bag` to stream the image data to the running ros node. 
7. Then, kill the ros node in third terminal, and run `python3 ros_single_object_tracking.py -s 3` to start the ros node for object tracking on sequence 3. 
8. In seventh terminal, run `rosbag play ./rosbags/seq3.bag` to stream the image data to the running ros node. 
