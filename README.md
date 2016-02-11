# deepRLNav

This a ros-python-c++ project for experimenting unsupervised obstacle navigation using deep networks. This has several packages namely,
<ul>
<li>map_and_state (ROS): Broadcasting map and the state of the robot</li>
<li>move_robot (ROS/C++): Send commands to the robot according to the given system arguments</li>
<li>ra_dae (python): Deep Autoencoder for learning from the images</li>
<li>conv_nets (python): Convolutional Nets for learning from the images</li>
<li>atrv_save_data.py (ROS/python): Broadcast images and laser information when robots moving</li>
<li>morse_scene_setup.py (ROS/python): Robot & Environment for the morse simulation</li>
<li>move_robot_exec.py (ROS/python): Moving the robot by subscribing to a topic which sends which action to take</li>
</ul>

<strong>Keywords: </strong> ROS, Python, C++<br/>
<strong>Dependencies: </strong> autonomy (ROS), amcl (ROS)

<h2>How to run</h2>
<ol>
<li><code>catkin_make</code> <strong>map_and_state</strong> and <strong>move_robot</strong></li>
<li>Start ROS using <code>roscore</code></li>
<li>Start morse using <code>morse run &lt;scene file&gt;</code></li>
<li>Start map_and_state using <code>roslaunch &lt;pkg_name&gt; &lt;launch_file_name&gt;</code></li>
<li>Start amcl using <code>roslaunch</code></li>
<li>Start rviz using <code>rosrun rviz rviz</code></li>
<li>Do initial pose estimate with <strong>rviz</strong> (If necessary)</li>
<li>Start autonomy package using <code>roslaunch</code></li>
</ol>

