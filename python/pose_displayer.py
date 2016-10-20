__author__ = 'thushv89'

import rospy
import numpy as np

from std_msgs.msg import Bool,Int16
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from os import listdir
from os.path import isfile, join
import os
import pickle
import time
import tf

if __name__=='__main__':
    rospy.init_node("pose_display_node")
    map_pose_pub = rospy.Publisher("/robot_map_pose",Marker, queue_size=10)
    now = rospy.Time.now()
    listener = tf.TransformListener()
    listener.waitForTransform('/robot_pose', '/map', rospy.Time(0), rospy.Duration(5.0))
    (trans,rot) = listener.lookupTransform('/robot_pose', '/map', rospy.Time(0))

    p_i=0
    while not rospy.is_shutdown():
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = p_i%250
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        marker.pose.position.x = trans[0]
        marker.pose.position.y = trans[1]
        marker.pose.position.z = trans[2]
        marker.pose.orientation.x = rot[0]
        marker.pose.orientation.y = rot[1]
        marker.pose.orientation.z = rot[2]
        marker.pose.orientation.w = rot[3]

        map_pose_pub.publish(marker)
        p_i+=1

        time.sleep(0.05)

    rospy.spin()