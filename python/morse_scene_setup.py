__author__ = 'thushv89'

from morse.builder import *
#import utils
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

cam_frequency = 10
camera = VideoCamera()
#camera.translate(x = -1.5, z= 0.9)
camera.translate(x = 0.7, z= 0.5)
camera.rotate(y = -0.0)
#camera.properties(cam_width=256,cam_height=192,cam_far=500,cam_near=2.15)
camera.properties(cam_width=320,cam_height=240,cam_focal=3.,cam_near=0.1,cam_far=500)
camera.frequency(cam_frequency)
atrv.append(camera)
camera.add_interface('ros',topic='/camera')

# for localization
scan = Hokuyo()
scan.translate(x=0.275, z=0.252)
atrv.append(scan)
scan.properties(Visible_arc = False)
scan.properties(laser_range = 30.0)
scan.properties(resolution = 0.2)
scan.properties(scan_window = 180.0) #angle of laser
scan.create_laser_arc()
scan.add_interface('ros', topic='/scan',frame_id="laser", child_frame_id="base_link")

laser_frequency = 20
# actual learning
obs_laser = Sick() # range: field: 180deg, 180 sample points
obs_laser.translate(x=0.5,z=0.252)
obs_laser.properties(Visible_arc = True)
obs_laser.properties(resolution = 1)
obs_laser.properties(scan_window = 120)
obs_laser.properties(laser_range = 4.0)
obs_laser.frequency(laser_frequency)
atrv.append(obs_laser)
obs_laser.add_interface('ros', topic='/obs_scan')

motion = MotionXYW()
atrv.append(motion)
motion.add_interface('ros', topic='/cmd_vel')

# Set the environment
env = Environment('indoors-1/indoor-1')
#env = Environment('laas/grande_salle')
