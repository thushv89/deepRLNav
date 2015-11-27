__author__ = 'thushv89'

from morse.builder import *

# A 'naked' PR2 robot to the scene
atrv = ATRV()
atrv.translate(x=2.5, y=3.2, z=0.0)

# An odometry sensor to get odometry information
odometry = Odometry()
atrv.append(odometry)
odometry.add_interface('ros', topic="/odom",frame_id="odom", child_frame_id="base_link")

# Keyboard control
keyboard = Keyboard()
atrv.append(keyboard)

camera = VideoCamera()
camera.translate(z= 0.6)
camera.frequency(2)
atrv.append(camera)
camera.add_interface('ros',topic='/camera')

scan = Hokuyo()
scan.translate(x=0.275, z=0.252)
atrv.append(scan)
scan.properties(Visible_arc = False)
scan.properties(laser_range = 30.0)
scan.properties(resolution = 1.0)
scan.properties(scan_window = 180.0)
scan.create_laser_arc()
scan.add_interface('ros', topic='/scan',frame_id="laser", child_frame_id="base_link")

obs_laser = Sick() # range: 30m, field: 180deg, 180 sample points
obs_laser.translate(x=0.5,z=0.252)
obs_laser.properties(Visible_arc = True)
obs_laser.properties(resolution = 45.0)
obs_laser.properties(scan_window = 90)
obs_laser.properties(laser_range = 3.0)
obs_laser.frequency(2)
atrv.append(obs_laser)
obs_laser.add_interface('ros', topic='/obs_scan')

motion = MotionXYW()
atrv.append(motion)
motion.add_interface('ros', topic='/cmd_vel')

# Set the environment
env = Environment('indoors-1/indoor-1')