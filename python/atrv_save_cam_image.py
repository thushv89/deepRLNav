#!/usr/bin/env python

__author__ = 'thushv89'

from pymorse import Morse
from morse.builder import *
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool,Int16
from rospy_tutorials.msg import Floats
import numpy as np
from PIL import Image as img
import math
from rospy.numpy_msg import numpy_msg
import sys
import scipy.misc as sm

def callback_cam(msg):

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
        img_mat_data = img_mat.getdata()
        # use if you wann check images data coming have correct data (images)
        sm.imsave('img'+str(1)+'.jpg', np.asarray(num_data,dtype='uint8').reshape(192,-1))
        cam_sub.unregister()

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


prevPose = None
data_status_pub = None
obstacle_status_pub = None
obstacle_msg_sent = False
initial_data = True
if __name__=='__main__':      
    currInputs = []
    currLabels = []

    rospy.init_node("save_data_node")
    #rate = rospy.Rate(1)
    data_status_pub = rospy.Publisher('data_sent_status', Int16, queue_size=10)
    sent_input_pub = rospy.Publisher('data_inputs', numpy_msg(Floats), queue_size=10)
    sent_label_pub = rospy.Publisher('data_labels', numpy_msg(Floats), queue_size=10)
    obstacle_status_pub = rospy.Publisher('obstacle_status',Bool, queue_size=10)
    
    cam_sub = rospy.Subscriber("/camera/image", Image, callback_cam)
    #rate.sleep()
    rospy.spin() # this will block untill you hit Ctrl+C
