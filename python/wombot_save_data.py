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
import threading
from os import system
import pickle
from multiprocessing import Pool
from copy import deepcopy
import tf

def callback_cam(msg):
    global reversing,isMoving,got_action
    global cam_skip,cam_count,img_seq_idx,curr_cam_data,save_pose_data
    global pose_logger,curr_ori,curr_pose
    global save_img_seq,image_dir
    global logger
    global episode,algo_episode
    global image_updown,unproc_cam_data
    
    #print "saw image, %i,%s,%s,%s"%(cam_count,got_action,isMoving,reversing)
    if image_dir is not None and ((got_action and isMoving) or reversing):
        if img_seq_idx%utils.IMG_SAVE_SKIP==0:
            
            if save_img_seq:
                curr_cam_data.append((msg.data,episode,img_seq_idx))
            try:
                #listener.waitForTransform("map","base_link",rospy.Time.now(),rospy.Duration(3.0));
                (curr_pose,curr_ori) = listener.lookupTransform("map", "base_link",rospy.Time(0));
                save_pose_data.append((episode,img_seq_idx,algo_episode,curr_pose+curr_ori))
                logger.info("save pose ..")
                #time.sleep(0.1)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                logger.info("No transform detected")
            
        img_seq_idx += 1
            
    cam_count += 1
    if cam_count%cam_skip != 0:
        return

    if isMoving and got_action and not reversing:
        
        #print "took image,%i,%s"%(cam_count,got_action)
        data = msg.data
        #print "height:%s, length:%s"%(rows,cols)
        #num_data = []     
        #num_data.append(int(0.2989 * ord(data[i]) + 0.5870 * ord(data[i+1]) + 0.1140 * ord(data[i+2])))
        unproc_cam_data.append(data)


def callback_laser(msg):
    global obstacle_msg_sent
    global currLabels,currInputs
    global reversing,isMoving,got_action,obstacle
    global laser_range_0,laser_range_1,laser_range_2
    global laser_skip,laser_count
    global save_img_seq, curr_cam_data,img_seq_idx,image_dir,save_pose_data
    global pose_logger, logger, curr_pose,curr_ori,algo_episode
    global reverse_lock

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

    #print "Laser %s,%s,%s"%(isMoving,got_action,reversing)
    if isMoving and got_action and not reversing:
        obstacle = False
	#print "took laser %s,%s,%s"%(got_action,isMoving,reversing)
        filtered_ranges = np.asarray(rangesNum)
        filtered_ranges[filtered_ranges<utils.NO_RETURN_THRESH] = 1000

        logger.debug("Min range recorded:%.4f at laser count: %s",np.min(filtered_ranges),len(currLabels))

        # middle part of laser [45:75]
        if np.min(filtered_ranges[laser_range_1[0]:laser_range_1[1]])<bump_thresh_1 or \
                        np.min(filtered_ranges[laser_range_0[0]:laser_range_0[1]])<bump_thresh_0_2 or \
                        np.min(filtered_ranges[laser_range_2[0]:laser_range_2[1]])<bump_thresh_0_2:

            obstacle = True
            logger.debug("setting Obstacle to True")
           
            currLabels.append(0)

            reverse_lock.acquire()
    	    cmd = "rosservice call /autonomy/path_follower/cancel_request"
    	    system(cmd)
    	    logger.debug("Called cancel path request ...\n")
            logger.debug('Reverse lock acquired ...')
            if not move_complete:
                logger.info("Posting 0 cmd_vel data ...")
                stop_robot()
                logger.info("Finished posting 0 cmd_vel data ...")
                logger.debug("Was still moving and bumped ...\n")
                logger.debug('Laser recorded min distance of: %.4f',np.min(filtered_ranges))
                import time
                import os
                for l in range(len(currLabels)):
                    currLabels[l] = 0

                
                logger.debug('Sending data as a ROS message ...\n')                
                save_data()

                #we need to save images at this point as well.
                #however since reverse data also needs to be included, it is done in
                #reverse_robot() method

                rospy.sleep(0.1)
                obstacle_status_pub.publish(True)

                isMoving = False
                #got_action=False
                reverse_robot()

                
            else:
                currInputs=[]
                currLabels=[]
            logger.debug('Releasing the reverse lock ...')
            reverse_lock.release()
	else:
	    currLabels.append(1)

def stop_robot():
    import time
    global cmd_vel_pub
    global logger
    logger.debug("Posting 0 cmd_vel data ....")
    for _ in range(100):
	rospy.sleep(0.01)
	twist_msg = Twist()
	twist_msg.linear = Vector3(0,0,0)
	twist_msg.angular = Vector3(0,0,0)
	cmd_vel_pub.publish(twist_msg)
    logger.debug("Finished posting 0 cmd_vel data ...\n")

def reverse_robot():
    global logger
    logger.info("Reversing Robot ...\n")
    import time
    global vel_lin_buffer,vel_ang_buffer,reversing
    global cmd_vel_pub,restored_bump_pub
    global curr_cam_data,save_pose_data,image_dir,save_img_seq
    reversing = True
    
    logger.debug("Posting cmd_vel messages backwards with a %.2f delay",utils.REVERSE_PUBLISH_DELAY)
    for l,a in zip(reversed(vel_lin_buffer),reversed(vel_ang_buffer)):
        lin_vec = Vector3(-l[0],-l[1],-l[2])
        ang_vec = Vector3(-a[0],-a[1],-a[2])
        twist_msg = Twist()
        twist_msg.linear = lin_vec
        twist_msg.angular = ang_vec
	rospy.sleep(utils.REVERSE_PUBLISH_DELAY)
        cmd_vel_pub.publish(twist_msg)
    # publish last twist message so the robot reverse a bit more
    for _ in range(5):
	rospy.sleep(utils.REVERSE_PUBLISH_DELAY)
	cmd_vel_pub.publish(twist_msg)

    logger.debug("Finished posting cmd_vel messages backwards ...\n")
    rospy.sleep(0.5)
    logger.debug("Posting zero cmd_vel messages with %.2f delay",utils.ZERO_VEL_PUBLISH_DELAY)
    for _ in range(100):
        rospy.sleep(utils.ZERO_VEL_PUBLISH_DELAY)
        twist_msg = Twist()
        twist_msg.linear = Vector3(0,0,0)
        twist_msg.angular = Vector3(0,0,0)

        cmd_vel_pub.publish(twist_msg)
    logger.debug("Finished posting zero cmd_vel messages ...\n")
    reversing = False
    restored_bump_pub.publish(True)

    logger.info("Reverse finished ...\n")
    
    #save_img_seq_thread = threading.Thread(target=save_img_sequence_pose_threading())
    #save_img_seq_thread.start()
    save_img_sequence_pose()
    logger.info('Saving images finished...')

# we use this call back to detect the first ever move after termination of move_exec_robot script
# after that we use callback_action_status
# the reason to prefer callback_action_status is that we can make small adjustments to robots pose without adding data
def callback_odom(msg):
    global prevPose
    global isMoving,initial_data
    global curr_pose,curr_ori,curr_cam_data
    global img_seq_idx,episode,algo_episode
    global no_move_count
    #if initial_data:
    
    
    data = msg
    pose = data.pose.pose # has position and orientation
    
    #try:
        #listener.waitForTransform("map","base_link",rospy.Time.now(),rospy.Duration(3.0));
        #(curr_pose,curr_ori) = listener.lookupTransform("map", "base_link",rospy.Time(0));
    #except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #    logger.info("No transform detected")
    #curr_pose = [float(pose.position.x),float(pose.position.y),float(pose.position.z)]
    #curr_ori = [float(pose.orientation.x),float(pose.orientation.y),float(pose.orientation.z),float(pose.orientation.w)]

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
            no_move_count += 1
            if(no_move_count>utils.NO_MOVE_COUNT_THRESH):
                #logger.debug('Set isMoving to False by odom')
                isMoving = False
        else:
            no_move_count=0
            isMoving = True
    else:
    	isMoving = False

    prevPose = data.pose.pose

def callback_path_finish(msg):
    from os import system
    import time
    global move_complete,isMoving,obstacle,got_action
    global cam_count,laser_count
    global save_img_seq,img_seq_idx,curr_cam_data,image_dir,save_pose_data
    global curr_pose,curr_ori
    global episode
    global logger,pose_logger
    global currInputs,currLabels

    logger.debug("Recieved callback_path_finish: %s",msg.data)
    time.sleep(0.05)
    if bool(msg.data)==True or (bool(msg.data)==False and not obstacle):
        logger.debug("msg.data: %s, obstacle: %s",msg.data,obstacle)
        logger.info("Sending data as a ROS message...\n")
        logger.debug("Set move_complete to True")

        if not obstacle:
            got_action = False
            move_complete=True
            
            save_data()
            #save_img_seq_thread = threading.Thread(target=save_img_sequence_pose_threading())
            #save_img_seq_thread.start()
            save_img_sequence_pose()
            logger.info("Saving images finished ...")
        else:
            logger.info('Reached end of path, but hit obstacle...\n')
            
        if isMoving:
            cmd = 'rosservice call /autonomy/path_follower/cancel_request'
            system(cmd)
            logger.info('Robot was still moving. Manually killing the path')

def save_img_sequence_pose():
    global image_dir,save_img_seq
    global curr_cam_data,save_pose_data
    global episode

    copy_cam_data = deepcopy(curr_cam_data)
    copy_pose_data = deepcopy(save_pose_data)
    curr_cam_data=[]
    save_pose_data=[]

    logger.info("Storage summary for episode %s",episode)
    logger.info('\tImage count: %s\n',len(copy_cam_data))
    logger.info("\tPose count: %s",len(copy_pose_data))
    pose_ep = copy_pose_data[0][0]
    if image_dir is not None:
        if save_img_seq:
            pool = Pool(utils.THREADS)
            pool.map(save_image, copy_cam_data)
            pool.close()
            pool.join()
                    
        pickle.dump(copy_pose_data,open(image_dir+os.sep+'pose_'+str(pose_ep)+'.pkl','wb'))
        
def save_img_sequence_pose_threading():
    global image_dir,save_img_seq
    global curr_cam_data,save_pose_data
    global episode

    copy_cam_data = deepcopy(curr_cam_data)
    copy_pose_data = deepcopy(save_pose_data)
    curr_cam_data=[]
    save_pose_data=[]

    pose_ep = copy_pose_data[0][0]
    if image_dir is not None:
        if save_img_seq:
            for cam in copy_cam_data:
                save_image(cam)
                    
        pickle.dump(copy_pose_data,open(image_dir+os.sep+'pose_'+str(pose_ep)+'.pkl','wb'))
        

def callback_action_status(msg):
    global isMoving, move_complete,got_action
    global vel_lin_buffer,vel_ang_buffer
    global img_seq_idx,cam_count,laser_count
    global episode,logger

    logger.info('Received Action message...')
    move_complete = False
    got_action = True
        #empty both velocity buffers
    vel_lin_buffer,vel_ang_buffer=[],[]

    img_seq_idx = 0
    cam_count,laser_count = 0,0
    episode += 1
    logger.info('Starting Episode: %s',episode)

def callback_algo_episode(msg):
    global algo_episode
    algo_episode = int(msg.data)

def callback_cmd_vel(msg):
    global vel_lin_buffer,vel_ang_buffer,isMoving,obstacle,reversing
    if isMoving and not reversing and not obstacle:
        lin_vec = msg.linear #Vector3 object
        ang_vec = msg.angular
        vel_lin_buffer.append([lin_vec.x,lin_vec.y,lin_vec.z])
        vel_ang_buffer.append([ang_vec.x,ang_vec.y,ang_vec.z])

def save_image(img_data_with_ep_seq):
    global image_updown

    img_data,ep,seq = img_data_with_ep_seq

    #print len(img_data),ep,seq
    img_np_2=np.empty((utils.IMG_H,utils.IMG_W,3),dtype=np.int16)
    for i in range(0,len(img_data),3):
        r_idx,c_idx=(i//3)//utils.IMG_W,(i//3)%utils.IMG_W
        img_np_2[r_idx,c_idx,(i%3)]=int(ord(img_data[i]))
        img_np_2[r_idx,c_idx,(i%3)+1]=int(ord(img_data[i+1]))
        img_np_2[r_idx,c_idx,(i%3)+2]=int(ord(img_data[i+2]))
    if image_updown:
        img_preprocessed_2 = np.fliplr(img_np_2)
        img_preprocessed_2 = np.flipud(img_preprocessed_2)

    sm.imsave(image_dir + os.sep + 'img_' +str(ep) + '_' + str(seq) + ".png",img_preprocessed_2)

def pre_proc_image(img_data,test_mode=False):
    global image_dir,episode,img_seq_idx
    img_np=np.empty((utils.IMG_H,utils.IMG_W),dtype=np.int16)

    for i in range(0,len(img_data),3):
        r_idx,c_idx=(i//3)//utils.IMG_W,(i//3)%utils.IMG_W
        img_np[r_idx,c_idx]=int(0.2989 * ord(img_data[i]) + 0.5870 * ord(img_data[i+1]) + 0.1140 * ord(img_data[i+2]))
    
    if test_mode:
        sm.imsave(image_dir + os.sep + 'test_img' +str(episode) + '_' + str(img_seq_idx) + ".png",img_np)
    #print "num_data:%s"%len(num_data)    
    resized_img_np = sm.imresize(img_np, (utils.THUMBNAIL_H,utils.THUMBNAIL_W), interp='nearest', mode=None)
    if test_mode:
        logger.debug("Resized IMG: %s(Res)",resized_img_np.shape)
        sm.imsave(image_dir + os.sep + 'test_img_resized' +str(episode) + '_' + str(img_seq_idx) + ".png",resized_img_np)
    #mat = np.reshape(np.asarray(num_data,dtype='uint8'),(rows,-1))
    #print "mat:h:%s,w:%s"%(mat.shape[0],mat.shape[1])
    #img_mat = img.fromarray(mat)
    #img_mat.thumbnail((thumbnail_w,thumbnail_h))
    #img_mat_data = img_mat.getdata()

    vertical_cut_threshold = int(thumbnail_h/5.)
    #img_preprocessed = np.reshape(np.asarray(list(img_mat_data)),(thumbnail_h,thumbnail_w))
    # use if you wann check images data coming have correct data
    #sm.imsave('img'+str(1)+'.jpg', img_data_reshape)

    resized_img_np = np.delete(resized_img_np,np.s_[:vertical_cut_threshold],0)
    resized_img_np = np.delete(resized_img_np,np.s_[-vertical_cut_threshold:],0)

    if test_mode:
        sm.imsave(image_dir + os.sep + 'test_img_v_cut' +str(episode) + '_' + str(img_seq_idx) + ".png",resized_img_np)
    if image_updown:
        img_preprocessed = np.fliplr(resized_img_np)
        img_preprocessed = np.flipud(img_preprocessed)
        if test_mode:
            sm.imsave(image_dir + os.sep + 'test_img_updown' +str(episode) + '_' + str(img_seq_idx) + ".png",img_preprocessed)

    # use if you wann check images data coming have correct data (image cropped resized)
    '''sm.imsave('avg_img'+str(1)+'.jpg', np.reshape(img_data_reshape,
                                                  (int(thumbnail_h-2*vertical_cut_threshold),-1))
              )'''
    return img_preprocessed.flatten()

def save_data():
    import time    
    
    global unproc_cam_data,currInputs,currLabels,initial_data
    global data_status_pub,sent_input_pub,sent_label_pub,logger
    global got_action

    pre_proc_pool = Pool(utils.THREADS)
    currInputs = pre_proc_pool.map(pre_proc_image, unproc_cam_data[len(unproc_cam_data)//2:])
    pre_proc_pool.close()
    pre_proc_pool.join()

    print "CurrInputs: %s(size) "%(len(currInputs))
    #for img in unproc_cam_data:

    #    currInputs.append(pre_proc_image(img))
    sent_input_pub.publish(np.asarray(currInputs,dtype=np.float32).reshape((-1,1)))
    logger.debug('Got %i data points (cam) but publishing only %i',len(unproc_cam_data),len(currInputs))
    sent_label_pub.publish(np.asarray(currLabels[len(currLabels)//2:],dtype=np.float32).reshape((-1,1)))
    logger.debug('Got %i data points (laser) but publishing only %i',len(currLabels),len(currLabels[len(currLabels)//2:]))
    time.sleep(0.1)
    if initial_data:
        data_status_pub.publish(0)
    else:
        data_status_pub.publish(1)

    logger.info("Data summary for episode %s",episode)
    logger.info('\tLaser count: %s\n',len(currLabels))
    logger.info("\tImage count: %s",len(currInputs))

    unproc_cam_data=[]
    currInputs=[]
    currLabels=[]
    initial_data = False
    got_action = False

logging_level = logging.DEBUG
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

thumbnail_w = utils.THUMBNAIL_W
thumbnail_h = utils.THUMBNAIL_H

reversing = False
isMoving = False
got_action = True
initial_data = True
move_complete = False
obstacle = False

no_move_count=0 # this is needed because we get 0 in between moves possibly due to higy broadcast frequency (50hz)

prevPose = None
data_status_pub = None
obstacle_status_pub = None
cmd_vel_pub = None
restored_bump_pub = None
obstacle_msg_sent = False
sent_input_pub = None

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
curr_cam_data = []
unproc_cam_data = []
save_pose_data = []
episode = 0
algo_episode = 0
image_updown = False

reverse_lock = threading.Lock()
save_data_thread = None
tf_lookup_thread = None
pool = None

listener = None
storing_img = False

from os import listdir
from os.path import isfile,join

if __name__=='__main__':

    global save_img_seq,image_dir,logger,pose_logger,logging_level,logging_format
    global episode
    global algo_episode

    import getopt
    import os.path

    
    

    logger = logging.getLogger('Logger')
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    logger.info('Laser topic: %s',utils.LASER_SCAN_TOPIC)
    logger.info('Camera topic: %s',utils.CAMERA_IMAGE_TOPIC)

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

    '''if image_dir and os.path.isfile(image_dir + os.sep + 'trajectory.log'):
        with open(image_dir + os.sep + 'trajectory.log') as f:
            f = f.readlines()
        img_seq_idx = int(f[-1].split(',')[1]) + 1
        episode = int(f[-1].split(',')[0]) + 1'''
    if image_dir and os.path.isfile(image_dir + os.sep + 'trajectory.log'):
	fnames = [f for f in listdir(image_dir) if isfile(join(image_dir,f)) and ".pkl" in f]
	max_f = 0
	for f in fnames:
	    f_ep = int(f.split('.')[0].split('_')[1])
	    if f_ep>max_f:
		max_f = f_ep
	episode = max_f+1

    if image_dir is not None:
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

    logger.info("Camera Frequency:%i",utils.CAMERA_FREQUENCY)
    logger.info("Laser Frequency:%i",utils.LASER_FREQUENCY)
    logger.info("Pref Frequency:%i",utils.PREF_FREQUENCY)
    logger.info("Skipping every %i camera frames",cam_skip-1)
    logger.info("Skipping every %i laser data",laser_skip-1)

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
    listener = tf.TransformListener()
    #tf_lookup_thread = threading.Thread(target=save_pose_via_lookup_tf())
    #tf_lookup_thread.start()

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
    rospy.Subscriber(utils.EPISODE_STATUS_TOPIC, Int16, callback_algo_episode)
    rospy.Subscriber(utils.CMD_VEL_TOPIC,Twist,callback_cmd_vel)

    #rate.sleep()
    rospy.spin() # this will block untill you hit Ctrl+C
