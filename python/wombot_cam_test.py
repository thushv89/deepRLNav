import rospy
from sensor_msgs.msg import Image
import numpy as np
from PIL import Image as img
import math
from rospy.numpy_msg import numpy_msg
import sys
import scipy.misc as sm
import time
def callback_cam(msg):

    global currInputs
    data = msg.data
    num_data = []
    rows = int(msg.height)
    cols = int(msg.width)
    print('h',rows,'w',cols)
    
    for i in range(0,len(data),4):
	num_data.append(int(0.2989 * ord(data[i]) + 0.5870 * ord(data[i+1]) + 0.1140 * ord(data[i+2])))
    print "num_data:%s"%len(num_data)
    mat = np.reshape(np.asarray(num_data,dtype='uint8'),(rows,-1))
    img_mat = img.fromarray(mat)
    img_mat_data = img_mat.getdata()
    
    num_data = np.asarray(num_data,dtype='uint8').reshape(480,-1)
    num_data = np.fliplr(num_data)
    num_data = np.flipud(num_data)
    # use if you wann check images data coming have correct data (images)
    sm.imsave('img'+str(1)+'.jpg',num_data )
    time.sleep(100)

if __name__=='__main__':      

    rospy.init_node("save_data_node")
    #rate = rospy.Rate(1)
    cam_sub = rospy.Subscriber("/camera/image_color", Image, callback_cam)
    #rate.sleep()
    rospy.spin() # this will block untill you hit Ctrl+C
