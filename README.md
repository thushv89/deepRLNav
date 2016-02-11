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
<strong>Dependencies: </strong> python 2.7, autonomy (ROS), amcl (ROS)

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
<li>run <code>atrv_save_data.py</code></li>
<li>run <code>ra_dae</code>(Train.py) or <code>conv_nets</code>(ConvNet.py)</li>
<li>run <code>move_robot_exec.py</code></li>
</ol>

<h2>How it works</h2>
<ol>
<li>Initiall, robot moves <code>step_size</code> straight</li>
<li>Once it starts moving, atrv_save_data.py will collect data until robot stops. This is done by listening to <code>/camera/image</code> and <code>/obs_scan</code> topics.</li>
<li>atrv_save_data.py will broadcast data on <code>/data_inputs</code> and <code>/data_labels</code> topics. And will broadcast True on <code>/data_sent_status</code> topic</li>
<li>Then, ra_dae or conv_net will be listening on <code>/data_inputs</code>, <code>/data_labels</code> and <code>/data_sent_status</code> topics and will start running the algorithm as soon as data_sent_status is True.</li>
<li>After running the algorithm for all the batches of data, once it is done, it will broadcast the output action on the topic <code>/action_status</code></li>
<li>Finally, move_robot_exec.py will receive the action from topic action_status and move the robot by <code>step_size</code> in the correct direction.</li>
<li>The process repeats. </li>
</ol>

<h2>Few important stuff</h2>
<ol>
<li>Changing the <code>step_size</code>: Variable available in the <code>move_robot.cpp</code> in move_robot package</li>
<li>Restoring learning algorithm parameters (only for ra_dae): When running the <code>ra_dae</code> set <code>--restore_last=1</code>. Data is stored to a .pkl file. It has <code>((W,b,b_prime) of layers),(Initial sizes of layers),Q values, last_episode</code> values in it.</li>
</ol>
