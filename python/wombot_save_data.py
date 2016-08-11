#!/usr/bin/env python

__author__ = 'thushv89'


import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool,Int16
from rospy_tutorials.msg import Floats
import numpy as np
from PIL import Image as img
import math
from rospy.numpy_msg import numpy_msg
import sys
import scipy.misc as sm
import utils
import logging

def save_img(data):
    global image_dir,episode,img_seq_idx
    rows = utils.IMG_H
    cols = utils.IMG_W
    img_mat = np.empty((rows,cols, 3), dtype=np.int32)

    j=0
    for i in range(0, len(data), 3):
        img_mat[j / cols, j % cols, 0] = int(ord(data[i]) * 256)
        img_mat[j / cols, j % cols, 1] = int(ord(data[i + 1]) * 256)
        img_mat[j / cols, j % cols, 2] = int(ord(data[i + 2]) * 256)
        j += 1
        # num_data.append(int(0.2989 * ord(data[i]) + 0.5870 * ord(data[i + 1]) + 0.1140 * ord(data[i + 2])))

    #image = img.fromarray(img_mat)
    sm.imsave(image_dir + os.sep + 'img' +str(episode) + '_' + str(img_seq_idx) + ".png",img_mat)

def callback_cam(msg):
    global reversing,isMoving,got_action
    global cam_skip,cam_count,img_seq_idx,curr_cam_data
    global pose_logger,curr_ori,curr_pose
    global save_img_seq
    global logger

    cam_count += 1
    if cam_count%cam_skip != 0:
        return

    if save_img_seq and (isMoving or reversing):

        if img_seq_idx%utils.IMG_SAVE_SKIP==0:
            save_img(msg.data)
            pose_logger.info("%s,%s,%s", episode,img_seq_idx, curr_pose + curr_ori)

        img_seq_idx += 1
        curr_cam_data = msg.data	

    
    if isMoving and not reversing:
        global currInputs,image_updown

        data = msg.data
        
        rows = int(msg.height)
        cols = int(msg.width)
        #print "height:%s, length:%s"%(rows,cols)
        num_data = []
        #print "data:%s"%len(data)        
        isAutoencoder = True
        if isAutoencoder:
            for i in range(0,len(data),3):
                num_data.append(int(0.2989 * ord(data[i]) + 0.5870 * ord(data[i+1]) + 0.1140 * ord(data[i+2])))
	    
            #print "num_data:%s"%len(num_data)	    
            mat = np.reshape(np.asarray(num_data,dtype='uint8'),(rows,-1))
            #print "mat:h:%s,w:%s"%(mat.shape[0],mat.shape[1])
            img_mat = img.fromarray(mat)
            img_mat.thumbnail((thumbnail_w,thumbnail_h))
            img_mat_data = img_mat.getdata()

            vertical_cut_threshold = int(thumbnail_h/5.)
            img_preprocessed = np.reshape(np.asarray(list(img_mat_data)),(thumbnail_h,thumbnail_w))
            # use if you wann check images data coming have correct data
            #sm.imsave('img'+str(1)+'.jpg', img_data_reshape)

            img_preprocessed = np.delete(img_preprocessed,np.s_[:vertical_cut_threshold],0)
            img_preprocessed = np.delete(img_preprocessed,np.s_[-vertical_cut_threshold:],0)

            if image_updown:
    	       img_preprocessed = np.fliplr(img_preprocessed)
    	       img_preprocessed = np.flipud(img_preprocessed)
            # use if you wann check images data coming have correct data (image cropped resized)
            '''sm.imsave('avg_img'+str(1)+'.jpg', np.reshape(img_data_reshape,
                                                          (int(thumbnail_h-2*vertical_cut_threshold),-1))
                      )'''
            img_preprocessed.flatten()

        currInputs.append(list(img_preprocessed))

def callback_laser(msg):
    global obstacle_msg_sent
    global currLabels,currInputs
    global reversing,isMoving,got_action
    global laser_range_0,laser_range_1,laser_range_2
    global laser_skip,laser_count
    global save_img_seq, curr_cam_data,img_seq_idx
    global pose_logger, logger, curr_pose,curr_ori

    laser_count += 1
    if laser_count % laser_skip != 0:
        return

    rangesTup = msg.ranges
    rangesNum = [float(r) for r in rangesTup]
    rangesNum.reverse()

    bump_thresh_1 = utils.BUMP_1_THRESH
    bump_thresh_0_2 = utils.BUMP_02_THRESH

    algo  = 'DeepRLMultiLogreg'
    #print(np.mean(rangesNum[0:15]),np.mean(rangesNum[45:75]),np.mean(rangesNum[105:120]))

    only_look_ahead = True
    if isMoving and not reversing:
        #print(rangesNum)
        labels = [0,0,0]
        obstacle = False

        filtered_ranges = np.asarray(rangesNum)
        filtered_ranges[filtered_ranges<utils.NO_RETURN_THRESH] = 1000

        logger.debug("Min range recorded:%.4f at laser count: %s",np.min(filtered_ranges),len(currLabels))

        if np.min(filtered_ranges[laser_range_1[0]:laser_range_1[1]])<bump_thresh_1 or \
                        np.min(filtered_ranges[laser_range_0[0]:laser_range_0[1]])<bump_thresh_0_2 or \
                        np.min(filtered_ranges[laser_range_2[0]:laser_range_2[1]])<bump_thresh_0_2:
            logger.info("Laser less than threshold. Obstacle set to True ...\n")
            obstacle = True

        if not obstacle:
            labels = [0,1,0]
        
        # if there is a one in labels
        if(1 in labels):
            currLabels.append(1)
        # if there is no 1 in labels
        else:
            currLabels.append(0)

        # middle part of laser [45:75]
        if np.min(filtered_ranges[laser_range_1[0]:laser_range_1[1]])<bump_thresh_1 or \
                        np.min(filtered_ranges[laser_range_0[0]:laser_range_0[1]])<bump_thresh_0_2 or \
                        np.min(filtered_ranges[laser_range_2[0]:laser_range_2[1]])<bump_thresh_0_2:
            if not move_complete:
                logger.debug("Was still moving and bumped ...\n")
                logger.debug('Laser recorded min distance of: %.4f',np.min(filtered_ranges))
                import time
                import os
                for l in range(len(currLabels)):
                    currLabels[l] = 0
                save_data()
                time.sleep(0.1)
                obstacle_status_pub.publish(True)

                if save_img_seq:
                    save_img(curr_cam_data)
                    pose_logger.info("%s,%s", img_seq_idx, curr_pose + curr_ori)
                    img_seq_idx += 1

                time.sleep(0.1)
                isMoving = False
                reverse_robot()

            else:
                currInputs=[]
                currLabels=[]

def reverse_robot():
    logger.info("Reversing Robot ...\n")
    import time
    global vel_lin_buffer,vel_ang_buffer,reversing
    global cmd_vel_pub,restored_bump_pub
    reversing = True
    # make sure robot comes to full stop
    time.sleep(0.5)
    logger.debug("Posting cmd_vel messages backwards with a %.2f delay",utils.REVERSE_PUBLISH_DELAY)
    for l,a in zip(reversed(vel_lin_buffer),reversed(vel_ang_buffer)):
        lin_vec = Vector3(-l[0],-l[1],-l[2])
        ang_vec = Vector3(-a[0],-a[1],-a[2])
        time.sleep(utils.REVERSE_PUBLISH_DELAY)
        twist_msg = Twist()
        twist_msg.linear = lin_vec
        twist_msg.angular = ang_vec
        cmd_vel_pub.publish(twist_msg)

    logger.debug("Posting zero cmd_vel messages with %.2f delay",utils.ZERO_VEL_PUBLISH_DELAY)
    for _ in range(10):
        time.sleep(utils.ZERO_VEL_PUBLISH_DELAY)
        twist_msg = Twist()
        twist_msg.linear = Vector3(0,0,0)
        twist_msg.angular = Vector3(0,0,0)

        cmd_vel_pub.publish(twist_msg)
    reversing = False
    restored_bump_pub.publish(True)
    logger.info("Reverse finished ...\n")

# we use this call back to detect the first ever move after termination of move_exec_robot script
# after that we use callback_action_status
# the reason to prefer callback_action_status is that we can make small adjustments to robots pose without adding data
def callback_odom(msg):
    global prevPose
    global isMoving,initial_data
    global curr_pose,curr_ori
    #if initial_data:
    
    data = msg
    pose = data.pose.pose # has position and orientation
     
    curr_pose = [float(pose.position.x),float(pose.position.y),float(pose.position.z)]
    curr_ori = [float(pose.orientation.x),float(pose.orientation.y),float(pose.orientation.z),float(pose.orientation.w)]

    if not reversing:
    	x = float(pose.position.x)
    	prevX = float(prevPose.position.x)  if not prevPose==None else 0.0
    	y = float(pose.position.y)
    	prevY =float(prevPose.position.y) if not prevPose==None  else 0.0
    	z = float(pose.position.z)
    	prevZ = float(prevPose.position.z) if not prevPose==None else 0.0

    	xo = float(pose.orientation.x)
    	prevXO = float(prevPose.orientation.x)  if not prevPose==None else 0.0
    	yo = float(pose.orientation.y)
    	prevYO =  float(prevPose.orientation.y) if not prevPose==None else 0.0
    	zo = float(pose.orientation.z)
    	prevZO =  float(prevPose.orientation.z) if not prevPose==None else 0.0
    	wo = float(pose.orientation.w)
    	prevWO = float(prevPose.orientation.w) if not prevPose==None  else 0.0

	#print "Pose:%s"%np.max([abs(x - prevX),abs(y - prevY),abs(z - prevZ)])
	#print "Orientation:%s\n"%np.max([abs(xo - prevXO),abs(yo - prevYO),abs(zo - prevZO),abs(wo - prevWO)])
    	pose_tol = 0.001
        ori_tol = 0.001
        if(abs(x - prevX)<pose_tol and abs(y - prevY)<pose_tol and abs(z - prevZ)<pose_tol
               and abs(xo - prevXO)<ori_tol and abs(yo - prevYO)<ori_tol and abs(zo - prevZO)<ori_tol and abs(wo - prevWO)<ori_tol):
            isMoving = False
        else:
            isMoving = True

    else:
    	isMoving = False
    prevPose = data.pose.pose

def callback_path_finish(msg):
    from os import system
    import time
    global move_complete,isMoving,obstacle
    global cam_count,laser_count
    global save_img_seq,img_seq_idx,curr_cam_data
    global curr_pose,curr_ori
    global episode
    global logger
    global currInputs,currLabels

    if int(msg.data)==1:
        logger.info("Sending data as a ROS message...\n")
        move_complete = True

        if not obstacle:
            save_data()
        else:
            logger.info('Reached end of path, but hit obstacle...\n')
            
        time.sleep(0.1)
        if isMoving:
            cmd = 'rosservice call /autonomy/path_follower/cancel_request'
            system(cmd)
            logger.info('Robot was still moving. Manually killing the path')

        if save_img_seq:
                save_img(curr_cam_data)
                pose_logger.info("%s,%s,%s",episode,img_seq_idx,curr_pose+curr_ori)

        logger.info("Data summary for episode %s",episode)
        logger.info("\tImage count: %s",len(currInputs))
        logger.info('\tLaser count: %s\n',len(currLabels))

def callback_action_status(msg):
    global isMoving, move_complete
    global vel_lin_buffer,vel_ang_buffer
    global img_seq_idx,cam_count,laser_count
    global episode
    global logger

    logger.info('Received Action message...')
    move_complete = False
    #isMoving = True
    #empty both velocity buffers
    vel_lin_buffer,vel_ang_buffer=[],[]

    img_seq_idx = 0
    cam_count,laser_count = 0,0
    episode += 1
    logger.info('Episode: %s',episode)

def callback_cmd_vel(msg):
    global vel_lin_buffer,vel_ang_buffer,isMoving
    if isMoving:
        lin_vec = msg.linear #Vector3 object
        ang_vec = msg.angular
        vel_lin_buffer.append([lin_vec.x,lin_vec.y,lin_vec.z])
        vel_ang_buffer.append([ang_vec.x,ang_vec.y,ang_vec.z])

def save_data():
    import time    
    
    global currInputs,currLabels,initial_data
    global data_status_pub,sent_input_pub,sent_label_pub

    sent_input_pub.publish(np.asarray(currInputs,dtype=np.float32).reshape((-1,1)))
    sent_label_pub.publish(np.asarray(currLabels,dtype=np.float32).reshape((-1,1)))
    time.sleep(0.1)
    if initial_data:
        data_status_pub.publish(0)
    else:
        data_status_pub.publish(1)

    currInputs=[]
    currLabels=[]
    initial_data = False

thumbnail_w = utils.THUMBNAIL_W
thumbnail_h = utils.THUMBNAIL_H

reversing = False
isMoving = False
got_action = False
initial_data = True
move_complete = False
obstacle = False

prevPose = None
data_status_pub = None
obstacle_status_pub = None
cmd_vel_pub = None
restored_bump_pub = None
obstacle_msg_sent = False

laser_range_0,laser_range_1,laser_range_2 = None,None,None
cam_skip, laser_skip = 0,0
cam_count,laser_count = 0,0

vel_lin_buffer = []
vel_ang_buffer = []

pose_logger = None
logger = None

curr_pose = None
curr_ori = None

save_img_seq = False
image_dir = None
img_seq_idx = 0
curr_cam_data = None
episode = 0
image_updown = False

if __name__=='__main__':

    global save_img_seq,image_dir,logger,pose_logger
    global episode

    import getopt
    import os.path

    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "", ['image_sequence=', "image_dir=", "image_updown="])
    except getopt.GetoptError as err:
        print '<filename>.py --image_sequence=<0or1> --image_dir=<dirname> --image_updown=<0or1>'
	print err        
	sys.exit(2)

    if len(opts) != 0:
        for opt, arg in opts:
            if opt == '--image_sequence':
                #logger.info('--image_sequence: %s', arg)
                save_img_seq = bool(int(arg))
            if opt == '--image_dir':
                #logger.info('--image_dir: %s', arg)
                image_dir = arg
            if opt == '--image_updown':
                image_updown = bool(int(arg))

    if image_dir and not os.path.exists(image_dir):
        os.mkdir(image_dir)

    if image_dir and os.path.isfile(image_dir + os.sep + 'trajectory.log'):
        with open(image_dir + os.sep + 'trajectory.log') as f:
            f = f.readlines()
        img_seq_idx = int(f[-1].split(',')[1]) + 1
        episode = int(f[-1].split(',')[0]) + 1

    if save_img_seq and image_dir is not None:
        pose_logger = logging.getLogger('TrajectoryLogger')
        pose_logger.setLevel(logging.INFO)
        fh = logging.FileHandler(image_dir+os.sep+'trajectory.log')
        fh.setFormatter(logging.Formatter('%(message)s'))
        fh.setLevel(logging.INFO)
        pose_logger.addHandler(fh)

    currInputs = []
    currLabels = []

    cam_skip = (utils.CAMERA_FREQUENCY/utils.PREF_FREQUENCY)
    laser_skip = (utils.LASER_FREQUENCY/utils.PREF_FREQUENCY)

    #laser range slicing algorithm
    if utils.LASER_ANGLE<=180:
        laser_slice = int(utils.LASER_POINT_COUNT/6.)
        laser_range_0 = (0,int(laser_slice))
        laser_range_1 = (int((utils.LASER_POINT_COUNT/2.)-laser_slice),int((utils.LASER_POINT_COUNT/2.)+laser_slice))
        laser_range_2 = (-int(laser_slice),-1)

    # if laser exceeds 180 degrees
    else:
        laser_slice = int(((utils.LASER_POINT_COUNT*1.0/utils.LASER_ANGLE)*180.)/6.)
        cutoff_angle_per_side = int((utils.LASER_ANGLE - 180)/2.)
        ignore_points_per_side = (utils.LASER_POINT_COUNT*1.0/utils.LASER_ANGLE)*cutoff_angle_per_side
        laser_range_0 = (int(ignore_points_per_side),int(laser_slice+ignore_points_per_side))
        laser_range_1 = (int((utils.LASER_POINT_COUNT/2.)-laser_slice),int((utils.LASER_POINT_COUNT/2.)+laser_slice))
        laser_range_2 = (-int(laser_slice-ignore_points_per_side),-int(ignore_points_per_side))

    logger.info("Laser slicing information")
    logger.info("Laser points: %d",utils.LASER_POINT_COUNT)
    logger.info("Laser angle: %d",utils.LASER_ANGLE)
    logger.info("Laser slice size: %d",laser_slice)
    logger.info("Laser ranges 0(%s),1(%s),2(%s)\n",laser_range_0,laser_range_1,laser_range_2)

    rospy.init_node("save_data_node")
    #rate = rospy.Rate(1)
    data_status_pub = rospy.Publisher(utils.DATA_SENT_STATUS, Int16, queue_size=10)
    sent_input_pub = rospy.Publisher(utils.DATA_INPUT_TOPIC, numpy_msg(Floats), queue_size=10)
    sent_label_pub = rospy.Publisher(utils.DATA_LABEL_TOPIC, numpy_msg(Floats), queue_size=10)
    obstacle_status_pub = rospy.Publisher(utils.OBSTACLE_STATUS_TOPIC,Bool, queue_size=10)
    cmd_vel_pub = rospy.Publisher(utils.CMD_VEL_TOPIC,Twist,queue_size=10)
    restored_bump_pub = rospy.Publisher(utils.RESTORE_AFTER_BUMP_TOPIC,Bool, queue_size=10)

    rospy.Subscriber(utils.CAMERA_IMAGE_TOPIC, Image, callback_cam)
    rospy.Subscriber(utils.LASER_SCAN_TOPIC, LaserScan, callback_laser)
    rospy.Subscriber(utils.ODOM_TOPIC, Odometry, callback_odom)
    rospy.Subscriber("/autonomy/path_follower_result",Bool,callback_path_finish)
    rospy.Subscriber(utils.ACTION_STATUS_TOPIC, Int16, callback_action_status)
    rospy.Subscriber(utils.CMD_VEL_TOPIC,Twist,callback_cmd_vel)

    #rate.sleep()
    rospy.spin() # this will block untill you hit Ctrl+C
