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
from multiprocessing import Pool

data_type = 'outdoor'

if data_type=='office':
    dir = 'test_right_pose'+os.sep+'test_office_ra_new'
    rviz_dir = 'test_right_pose'+os.sep+'test_rviz_office_ra_new'
    img_eps = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 374, 375, 376, 377, 378, 379, 380, 381]
elif data_type =='outdoor':
    dir = 'test_right_pose/outdoor'+os.sep+'test_outdoor_ra_new'
    rviz_dir = 'test_right_pose'+os.sep+'test_rviz_outdoor_ra_new'
    img_eps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305]
data_type ='pkl' #pkl or log
pose_arr = []
unproc_image_data = []
ep_id = 0
img_seq_id = 0
episodes = []
img_seq_indices = []
published_pose = False
ep_to_algo_ep_map = {}
prev_e = 0


def callback_img(msg):
    global published_pose,ep_id,img_seq_id
    if published_pose:
        #print 'Img added ',ep_id,img_seq_id
        unproc_image_data.append((ep_id,img_seq_id,msg.data))
        published_pose = False

def save_image(img_data_with_ep_seq):
    global image_updown

    ep,seq,img_data = img_data_with_ep_seq

    #print len(img_data),ep,seq
    img_np_2=np.empty((300,300,3),dtype=np.int16)
    for i in range(0,len(img_data),4):
        r_idx,c_idx=(i//4)//300,(i//4)%300
        img_np_2[r_idx,c_idx,(i%4)]=int(ord(img_data[i+2]))
        img_np_2[r_idx,c_idx,(i%4)+1]=int(ord(img_data[i+1]))
        img_np_2[r_idx,c_idx,(i%4)+2]=int(ord(img_data[i]))

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
                    px,py,pz,ox,oy,oz,ow = point[3][0],point[3][1],point[3][2],point[3][3],point[3][4],point[3][5],point[3][6]
                    if point[0]==11:
                        px+=0.1
                    if point[0]==12:
                        px+=0.2
                    if point[0]==13 or point[0]==14:
                        px+=0.3
                    if point[0]==376 or point[0]==270:
                        px-=0.1
                    episodes.append(point[0])
                    img_seq_indices.append(point[1])
                    pose_arr.append([px,py,pz,ox,oy,oz,ow])
                    ep_to_algo_ep_map[point[0]]=point[2]

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
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    arrow_marker = Marker()

    arrow_marker.header.frame_id = "map"
    arrow_marker.id = 10000

    arrow_marker.type = marker.ARROW
    arrow_marker.action = marker.ADD
    arrow_marker.scale.x = 0.25
    arrow_marker.scale.y = 0.15
    arrow_marker.scale.z = 0.01
    arrow_marker.color.a = 1.0
    arrow_marker.color.r = 1.0
    arrow_marker.color.g = 1.0
    arrow_marker.color.b = 0.0

    for p_i,(e,ims,pose) in enumerate(zip(episodes,img_seq_indices,pose_arr)):

        if e not in img_eps:
            continue

        while published_pose:
            True

        if np.abs(prev_e-e)>10:
                marker.points = []
                print "blank"
                rospy.sleep(1)

        if marker_type is 'sphere':
            marker = Marker()

            marker.header.frame_id = "map"
            marker.id = p_i

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

        if len(unproc_image_data)>500:
            pool = Pool(50)
            pool.map(save_image, unproc_image_data)
            pool.close()
            pool.join()
            unproc_image_data = []

        pose_pub.publish(marker)
        pose_arrow_pub.publish(arrow_marker)
        print 'pose added ',ep_id,img_seq_id

        published_pose = True
        #br.sendTransform((pose[0],pose[1],pose[2]),(pose[3],pose[4],pose[5],pose[6]),rospy.Time.now(),"map","/robot_pose")

        prev_e = e
        rospy.sleep(0.1)



    for img in unproc_image_data:
        save_image(img)
    rospy.spin()

