__author__ = 'thushv89'

import rospy
import numpy as np
from std_msgs.msg import Bool,Int16
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped,Point
from sensor_msgs.msg import Image
from os import listdir
from os.path import isfile, join
import os
import pickle
import time
import scipy.misc as sm

dir = 'test_right_pose'+os.sep+'test_office_ra_new'
rviz_dir = 'test_right_pose'+os.sep+'test_rviz_office_ra_new'
data_type ='pkl' #pkl or log
pose_arr = []
unproc_image_data = []
ep_id = 0
img_seq_id = 0
episodes = []
img_seq_indices = []
published_pose = False
def callback_img(msg):
    global published_pose,ep_id,img_seq_id
    data = msg.data
    num_data=[]
    if published_pose:
        #unproc_image_data.append((ep_id,img_seq_id,msg.data))
        published_pose = False

def save_image(img_data_with_ep_seq):
    global image_updown

    ep,seq,img_data = img_data_with_ep_seq

    #print len(img_data),ep,seq
    img_np_2=np.empty((300,300,3),dtype=np.int16)
    for i in range(0,len(img_data),4):
        r_idx,c_idx=(i//4)//300,(i//4)%300
        img_np_2[r_idx,c_idx,(i%4)]=int(ord(img_data[i]))
        img_np_2[r_idx,c_idx,(i%4)+1]=int(ord(img_data[i+1]))
        img_np_2[r_idx,c_idx,(i%4)+2]=int(ord(img_data[i+2]))

    sm.imsave(rviz_dir + os.sep + 'rviz_' +str(ep) + '_' + str(seq) + ".png",img_np_2)


if __name__=='__main__':
    global ep_id,img_seq_id
    global episodes,img_seq_indices
    global published_pose
    global unproc_image_data

    if data_type=='pkl':
        pkl_names = [f for f in listdir(dir) if isfile(join(dir, f)) and '.pkl' in f]
        max_pkl_idx = 0
        for fn in pkl_names:
            pkl_idx = int(fn.split('.')[0].split('_')[1])
            if pkl_idx>max_pkl_idx:
                max_pkl_idx=pkl_idx

        for i in range(max_pkl_idx):

            with open(dir+os.sep+'pose_'+str(i)+'.pkl', 'rb') as f:
                data = pickle.load(f)
                for point in data:
                    episodes.append(point[0])
                    img_seq_indices.append(point[2])
                    pose_arr.append(point[3])

        print(len(pose_arr))
    elif data_type=='log':
        raise NotImplementedError

    rospy.init_node("pose_node")
    pose_pub = rospy.Publisher("/robot_pose",Marker, queue_size=10)

    rviz_image_subscriber = rospy.Subscriber('/image',Image,callback_img)
    pose_arrow_pub =  rospy.Publisher("/robot_arrow_pose",Marker, queue_size=10)

    prev_x,prev_y,prev_z = pose_arr[0][0],pose_arr[0][1],pose_arr[0][2]
    marker_type = 'points' #['sphere','line_strip']

    marker = Marker()

    marker.header.frame_id = "map"
    marker.id = 0

    marker.action = marker.ADD
    marker.scale.x = 0.02
    marker.scale.y = 0.02
    marker.scale.z = 0.02
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    arrow_marker = Marker()

    arrow_marker.header.frame_id = "map"
    arrow_marker.id = 10000

    arrow_marker.type = marker.ARROW
    arrow_marker.action = marker.ADD
    arrow_marker.scale.x = 0.1
    arrow_marker.scale.y = 0.1
    arrow_marker.scale.z = 0.1
    arrow_marker.color.a = 1.0
    arrow_marker.color.r = 1.0
    arrow_marker.color.g = 1.0
    arrow_marker.color.b = 0.0

    for p_i,(e,ims,pose) in enumerate(zip(episodes,img_seq_indices,pose_arr)):
        if p_i==len(pose_arr):
            break
        while published_pose:
            True

        if marker_type is 'sphere':
            marker = Marker()

            marker.header.frame_id = "map"
            marker.id = (p_i)%250

            marker.action = marker.ADD
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker.type = marker.SPHERE
            marker.pose.position.x = pose[0]
            marker.pose.position.y = pose[1]
            marker.pose.position.z = pose[2]
            marker.pose.orientation.x = pose[3]
            marker.pose.orientation.y = pose[4]
            marker.pose.orientation.z = pose[5]
            marker.pose.orientation.w = pose[6]

        elif marker_type is 'line_strip' or marker_type is 'points':
            if marker_type is 'line_strip':
                marker.type = marker.LINE_STRIP
            elif marker_type is 'points':
                marker.type = marker.POINTS
            point = Point()
            point.x = pose[0]
            point.y = pose[1]
            point.z = pose[2]
            threshold = 1
            #if p_i>0 and (np.abs(pose_arr[p_i-1][0]-pose[0])>threshold or np.abs(pose_arr[p_i-1][1]-pose[1])>threshold or np.abs(pose_arr[p_i-1][2]-pose[2])>threshold):
            #    marker.points = []
            marker.points.append(point)


        arrow_marker.pose.position.x = pose[0]
        arrow_marker.pose.position.y = pose[1]
        arrow_marker.pose.position.z = pose[2]
        arrow_marker.pose.orientation.x = pose[3]
        arrow_marker.pose.orientation.y = pose[4]
        arrow_marker.pose.orientation.z = pose[5]
        arrow_marker.pose.orientation.w = pose[6]

        ep_id,img_seq_id = e,ims

        pose_pub.publish(marker)
        pose_arrow_pub.publish(arrow_marker)
        published_pose = True
        #br.sendTransform((pose[0],pose[1],pose[2]),(pose[3],pose[4],pose[5],pose[6]),rospy.Time.now(),"map","/robot_pose")

        prev_x,prev_y,prev_z = pose[0],pose[1],pose[2]
        rospy.sleep(0.05)

    #for img in unproc_image_data:
        #save_image(img)
    rospy.spin()

