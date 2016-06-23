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

def callback_cam(msg):
    global reversing
    global cam_skip,cam_count

    cam_count += 1
    if cam_count%cam_skip != 0:
        return


    
    if isMoving and not reversing:
        global currInputs

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

	    img_preprocessed = np.fliplr(img_preprocessed)
	    img_preprocessed = np.flipud(img_preprocessed)
            # use if you wann check images data coming have correct data (image cropped resized)
            '''sm.imsave('avg_img'+str(1)+'.jpg', np.reshape(img_data_reshape,
                                                          (int(thumbnail_h-2*vertical_cut_threshold),-1))
                      )'''
            img_preprocessed.flatten()
                     
        currInputs.append(list(img_preprocessed))
        print 'Input count: %s\n'%len(currInputs)

def callback_laser(msg):
    global obstacle_msg_sent
    global currLabels,currInputs
    global reversing,isMoving
    global laser_range_0,laser_range_1,laser_range_2
    global laser_skip,laser_count

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

        print "min range:%s"%np.min(filtered_ranges)

        if np.min(filtered_ranges[laser_range_1[0]:laser_range_1[1]])<bump_thresh_1 or \
                        np.min(filtered_ranges[laser_range_0[0]:laser_range_0[1]])<bump_thresh_0_2 or \
                        np.min(filtered_ranges[laser_range_2[0]:laser_range_2[1]])<bump_thresh_0_2:
            print "Obstacle set to True\n"
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
                print "Was still moving and bumped\n"
                import time
                import os
                for l in range(len(currLabels)):
                    currLabels[l] = 0
                save_data()
                time.sleep(0.1)
                obstacle_status_pub.publish(True)
                time.sleep(0.1)
                isMoving = False
                reverse_robot()

            else:
                currInputs=[]
                currLabels=[]

        print "Labels count: %s\n"%len(currLabels)

def reverse_robot():
    print "Reversing Robot\n"
    import time
    global vel_lin_buffer,vel_ang_buffer,reversing
    global cmd_vel_pub,restored_bump_pub
    reversing = True
    for l,a in zip(reversed(vel_lin_buffer),reversed(vel_ang_buffer)):
        lin_vec = Vector3(-l[0],-l[1],-l[2])
        ang_vec = Vector3(-a[0],-a[1],-a[2])
        time.sleep(utils.REVERSE_PUBLISH_DELAY)
        twist_msg = Twist()
        twist_msg.linear = lin_vec
        twist_msg.angular = ang_vec
        cmd_vel_pub.publish(twist_msg)

    for _ in range(10):
        time.sleep(utils.ZERO_VEL_PUBLISH_DELAY)
        twist_msg = Twist()
        twist_msg.linear = Vector3(0,0,0)
        twist_msg.angular = Vector3(0,0,0)

        cmd_vel_pub.publish(twist_msg)
    reversing = False
    restored_bump_pub.publish(True)


# we use this call back to detect the first ever move after termination of move_exec_robot script
# after that we use callback_action_status
# the reason to prefer callback_action_status is that we can make small adjustments to robots pose without adding data
def callback_odom(msg):
    global prevPose
    global isMoving,initial_data

    #if initial_data:
    
    data = msg
    pose = data.pose.pose # has position and orientation
    
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
    	tolerance = 0.001
    	if(abs(x - prevX)<tolerance and abs(y - prevY)<tolerance and abs(z - prevZ)<tolerance  and abs(xo - prevXO)<tolerance and abs(yo - prevYO)<tolerance and abs(zo - prevZO)<tolerance and abs(wo - prevWO)<tolerance):
	    isMoving = False
    	else:
	    
	    isMoving = True
    else:
    	isMoving = False
    prevPose = data.pose.pose

def callback_path_finish(msg):
    from os import system
    import time
    global move_complete,isMoving
    global cam_count,laser_count
    if int(msg.data)==1:
        print "saving data...\n"
        save_data()
        move_complete = True
        time.sleep(0.1)
        if isMoving:
            cmd = 'rosservice call /autonomy/path_follower/cancel_request'
            system(cmd)
            print 'Robot was still moving. Manually killing the path'

    cam_count,laser_count = 0,0

def callback_action_status(msg):
    global isMoving, move_complete
    global vel_lin_buffer,vel_ang_buffer
    move_complete = False
    #isMoving = True
    #empty both velocity buffers
    vel_lin_buffer,vel_ang_buffer=[],[]

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

if __name__=='__main__':      
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

    print "Laser slicing information"
    print "Laser points: %d" %utils.LASER_POINT_COUNT
    print "Laser angle: %d" %utils.LASER_ANGLE
    print "Laser slice size: %d" %laser_slice
    print "Laser ranges 0(%s),1(%s),2(%s)" %(laser_range_0,laser_range_1,laser_range_2)

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
