# SLAM in ROS

### Quick run
```bash
# source the ros environment
$ . /opt/ros/noetic/setup.bash

# source the cartographer environment
$ . <cartographer-catkin-ws>/install_isolated/setup.bash

$ cd <Homework-2>/task2/catkin_ws

$ catkin_make

# source the task2 workspace
$ . devel/setup.bash

$ roslaunch me5413_hw2_mapping me5413_hw2_mapping.launch bag_filename:=<absolute path to rosbag> record_bag_filename:=<absolute path to rosbag to be recorded>
```

After the mapping completes, in another terminal perform the following to save the map:
```bash
# source the ros environment
$ . /opt/ros/noetic/setup.bash

$ rosrun map_server map_saver --occ 70 --free 30 -f <Homework-2>/task2/results/task2 map:=/map
```

To evaluate the result:
```bash
# source the ros environment
$ . /opt/ros/noetic/setup.bash

$ evo_ape bag <path to results.bag> /ground_truth /tf:map.base_link --plot -a
```
