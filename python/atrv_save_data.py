#!/usr/bin/env python

__author__ = 'thushv89'

from pymorse import Morse
from morse.builder import *
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

def callback_cam(msg):
    global reversing
    if isMoving and not reversing:
        global currInputs

        data = msg.data
        
        rows = int(msg.height)
        cols = int(msg.width)
        print('h',rows,'w',cols)
        num_data = []
        
        isAutoencoder = True
        if isAutoencoder:
            for i in range(0,len(data),4):
                num_data.append(int(0.2989 * ord(data[i]) + 0.5870 * ord(data[i+1]) + 0.1140 * ord(data[i+2])))
	    
            mat = np.reshape(np.asarray(num_data,dtype='uint8'),(rows,-1))
            img_mat = img.fromarray(mat)
            img_mat.thumbnail((84,63))
            img_mat_data = img_mat.getdata()
            # use if you wann check images data coming have correct data (images)
            #sm.imsave('img'+str(1)+'.jpg', np.asarray(num_data,dtype='uint8').reshape(256,-1))
            #m.imsave('avg_img'+str(1)+'.jpg', np.asarray(list(img_mat_data)).reshape(64,-1))

        else:
            data_r = []
            data_g = []
            data_b = []
            for i in range(0,len(data),4):
                data_r.append(ord(data[i]))
                data_g.append(ord(data[i+1]))
                data_b.append(ord(data[i+2]))
            
            num_data.extend(data_r)
            num_data.extend(data_g)
            num_data.extend(data_b)
            
            #mat = np.reshape(np.asarray(num_data,dtype='uint8'),(rows,cols,-1))

            #img_mat = img.fromarray(mat,'RGB')

            #for RGB mode, img.getdata returns a 256x256 long array. each element is a 3 item tuple
            
            img_mat_data = num_data
            print('num data',len(num_data))
                     
        currInputs.append(list(img_mat_data))


def callback_laser(msg):
    global obstacle_msg_sent
    global currLabels,currInputs
    global reversing,isMoving

    rangesTup = msg.ranges
    rangesNum = [float(r) for r in rangesTup]
    rangesNum.reverse()
    #print "%.3f,%s" % (np.min(rangesNum),np.argmin(rangesNum))

    bump_thresh_1 = 0.6
    bump_thresh_0_2 = 0.75
    algo  = 'DeepRLMultiLogreg'
    #print(np.mean(rangesNum[0:15]),np.mean(rangesNum[45:75]),np.mean(rangesNum[105:120]))

    only_look_ahead = True
    if isMoving and not reversing:
        #print(rangesNum)
        labels = [0,0,0]
        obstacle = False

        if np.min(rangesNum[45:75])<bump_thresh_1 or np.min(rangesNum[0:45])<bump_thresh_0_2 or np.min(rangesNum[75:])<bump_thresh_0_2:
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
        if np.min(rangesNum[45:75])<bump_thresh_1 or np.min(rangesNum[0:45])<bump_thresh_0_2 or np.min(rangesNum[75:])<bump_thresh_0_2:
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

        print(currLabels)
    
def reverse_robot():
    print "Reversing Robot\n"
    import time
    global vel_lin_buffer,vel_ang_buffer,reversing
    global cmd_vel_pub,restored_bump_pub
    reversing = True
    for l,a in zip(reversed(vel_lin_buffer),reversed(vel_ang_buffer)):
        lin_vec = Vector3(-l[0],-l[1],-l[2])
        ang_vec = Vector3(-a[0],-a[1],-a[2])
        time.sleep(0.1)
        twist_msg = Twist()
        twist_msg.linear = lin_vec
        twist_msg.angular = ang_vec
        cmd_vel_pub.publish(twist_msg)

    for _ in range(10):
        time.sleep(0.05)
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

    if initial_data:
        data = msg
        pose = data.pose.pose # has position and orientation

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

        tolerance = 0.001
        if(abs(x - prevX)<tolerance and abs(y - prevY)<tolerance and abs(z - prevZ)<tolerance
           and abs(xo - prevXO)<tolerance and abs(yo - prevYO)<tolerance and abs(zo - prevZO)<tolerance and abs(wo - prevWO)<tolerance):
            isMoving = False
        else:
            isMoving = True

        prevPose = data.pose.pose

def callback_path_finish(msg):
    from os import system
    import time
    global move_complete,isMoving
    if int(msg.data)==1:
        print "saving data...\n"
        save_data()
        move_complete = True
        time.sleep(0.1)
        if isMoving:
            cmd = 'rosservice call /autonomy/path_follower/cancel_request'
            system(cmd)
            print 'Robot was still moving. Manually killing the path'
        isMoving = False

def callback_action_status(msg):
    global isMoving, move_complete
    global vel_lin_buffer,vel_ang_buffer
    move_complete = False
    isMoving = True
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

vel_lin_buffer = []
vel_ang_buffer = []

if __name__=='__main__':      
    currInputs = []
    currLabels = []

    rospy.init_node("save_data_node")
    #rate = rospy.Rate(1)
    data_status_pub = rospy.Publisher('data_sent_status', Int16, queue_size=10)
    sent_input_pub = rospy.Publisher('data_inputs', numpy_msg(Floats), queue_size=10)
    sent_label_pub = rospy.Publisher('data_labels', numpy_msg(Floats), queue_size=10)
    obstacle_status_pub = rospy.Publisher('obstacle_status',Bool, queue_size=10)
    cmd_vel_pub = rospy.Publisher('cmd_vel',Twist,queue_size=10)
    restored_bump_pub = rospy.Publisher('restored_bump',Bool, queue_size=10)

    rospy.Subscriber("/camera/image", Image, callback_cam)
    rospy.Subscriber("/obs_scan", LaserScan, callback_laser)
    rospy.Subscriber("/odom", Odometry, callback_odom)
    rospy.Subscriber("/autonomy/path_follower_result",Bool,callback_path_finish)
    rospy.Subscriber("/action_status", Int16, callback_action_status)
    rospy.Subscriber('/cmd_vel',Twist,callback_cmd_vel)

    #rate.sleep()
    rospy.spin() # this will block untill you hit Ctrl+C
