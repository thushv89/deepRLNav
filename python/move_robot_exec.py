import rospy
from std_msgs.msg import Int16
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
import utils

def callback_action_status(msg):
    import os
    cmd = utils.MOVE_ROBOT_CMD + str(msg.data)
    os.system(cmd)

def callback_obstacle_status(msg):
    global execute_actions
    execute_actions = False
    print("[move_robot_executer] Hit obstacle\n")
    import os
    #import time
    os.system(utils.GOAL_CANCEL_CMD)

def callback_restored_bump(msg):
    global execute_actions
    execute_actions=True

def callback_goal(msg):
    global initial_run,init_run_pub
    if initial_run:
        init_run_pub.publish(True)
        print('Initial run \n')
        initial_run = False

execute_actions = True
initial_run = True
init_run_pub = None
if __name__=='__main__':
    
    import os

    rospy.init_node("move_robot_executer")        
    rospy.Subscriber(utils.ACTION_STATUS_TOPIC, Int16, callback_action_status)
    rospy.Subscriber(utils.OBSTACLE_STATUS_TOPIC, Bool, callback_obstacle_status)
    rospy.Subscriber(utils.RESTORE_AFTER_BUMP_TOPIC, Bool, callback_restored_bump)
    rospy.Subscriber(utils.GOAL_TOPIC,PoseStamped,callback_goal)
    init_run_pub = rospy.Publisher(utils.INITIAL_RUN_TOPIC,Bool,queue_size=10)

    cmd = utils.MOVE_ROBOT_CMD + '1'
    os.system(cmd)

    rospy.spin()


