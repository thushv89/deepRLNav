#!/usr/bin/env python

__author__ = 'thushv89'

from pymorse import Morse
from morse.builder import *
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from rospy_tutorials.msg import Floats
import numpy as np
from PIL import Image as img
import math
from rospy.numpy_msg import numpy_msg
import sys
import scipy.misc as sm

def callback_cam(msg):
    if isMoving:
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
            img_mat.thumbnail((64,64))
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
        
        
        print('IMG SUCCESS')
    

def callback_laser(msg):
    global obstacle_msg_sent
    global currLabels
    rangesTup = msg.ranges
    rangesNum = [float(r) for r in rangesTup]
    rangesNum.reverse()
    min_range = min(rangesNum)
    bump_thresh = 0.8
    algo  = 'DeepRLMultiLogreg'
    #print(np.mean(rangesNum[0:15]),np.mean(rangesNum[45:75]),np.mean(rangesNum[105:120]))
    only_look_ahead = True
    if isMoving:
        #print(rangesNum)
        labels = [0,0,0]
        obstacle = False
        for l in rangesNum[45:75]:
            if l < bump_thresh:
                obstacle = True
        #if only_look_ahead or (l>bump_thresh/2 for l in rangesNum[0:15]):
        #    labels[0] = 0
        #if (l<bump_thresh for l in rangesNum[45:75]):
        #    labels[1] = 1
        #if only_look_ahead or (l>bump_thresh/2 for l in rangesNum[105:120]):
        #    labels[2] = 0
        if not obstacle:
            labels = [0,1,0]
        print(labels)            
        
        #idx_of_1 = [i for i,val in enumerate(labels) if val==1] #indexes which has 1 as value
        # if there are more than one 1 choose one randomly
        #while(len(idx_of_1)>=2):
        #    idx_of_1 = [i for i,val in enumerate(labels) if val==1]
        #    import random
        #    rand_idx = random.randint(0,len(idx_of_1)-1)
        #    labels[idx_of_1[rand_idx]]=0.0
        #    del idx_of_1[rand_idx]

        # if there is a one in labels
        if(1 in labels):
            currLabels.append(1)
        # if there is no 1 in labels
        else:
            currLabels.append(0)

        if np.min(rangesNum[45:75])<0.3:
            import time
            import os
            save_data()
            time.sleep(0.1)
            obstacle_status_pub.publish(True)
            time.sleep(0.5)
            cmd = 'rosnode kill /save_data_node'
            os.system(cmd)
            
        print(currLabels)
        print('Laser SUCCESS')    
    

def callback_odom(msg):
    global prevPose
    global isMoving
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
    
    #print("Moving status: ",isMoving)
    prevPose = data.pose.pose

def callback_path_finish(msg):
    if int(msg.data)==1:
        save_data()
  
def save_data():
    import time    
    
    global currInputs
    global currLabels
    global data_status_pub
    global sent_input_pub
    global sent_label_pub    
    #print("Input size: ",np.asarray(currInputs,dtype=np.float32).shape)
    #print("Label size: ",np.asarray(currLabels,dtype=np.float32).shape)
    sent_input_pub.publish(np.asarray(currInputs,dtype=np.float32).reshape((-1,1)))
    sent_label_pub.publish(np.asarray(currLabels,dtype=np.float32).reshape((-1,1)))
    time.sleep(0.1)
    data_status_pub.publish(True)
    currInputs=[]
    currLabels=[]

isMoving = False
prevPose = None
data_status_pub = None
obstacle_status_pub = None
obstacle_msg_sent = False
if __name__=='__main__':      
    currInputs = []
    currLabels = []

    rospy.init_node("save_data_node")
    #rate = rospy.Rate(1)
    data_status_pub = rospy.Publisher('data_sent_status', Bool, queue_size=10)
    sent_input_pub = rospy.Publisher('data_inputs', numpy_msg(Floats), queue_size=10)
    sent_label_pub = rospy.Publisher('data_labels', numpy_msg(Floats), queue_size=10)
    obstacle_status_pub = rospy.Publisher('obstacle_status',Bool, queue_size=10)
    
    rospy.Subscriber("/camera/image", Image, callback_cam)
    rospy.Subscriber("/obs_scan", LaserScan, callback_laser)
    rospy.Subscriber("/odom", Odometry, callback_odom)
    rospy.Subscriber("/autonomy/path_follower_result",Bool,callback_path_finish)
    #rate.sleep()
    rospy.spin() # this will block untill you hit Ctrl+C
