import rospy
from std_msgs.msg import Int16
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

def callback_action_status(msg):
    import os
    cmd = '../ros/move_robot/devel/lib/move_robot/move_robot ' + str(msg.data)
    os.system(cmd)

def callback_obstacle_status(msg):
    global execute_actions
    execute_actions = False
    print("[move_robot_executer] Hit obstacle\n")
    import os
    #import time
    cmd = 'rosservice call /autonomy/path_follower/cancel_request'
    os.system(cmd)

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
    rospy.Subscriber("/action_status", Int16, callback_action_status)
    rospy.Subscriber("/obstacle_status", Bool, callback_obstacle_status)
    rospy.Subscriber("/restored_bump", Bool, callback_restored_bump)
    rospy.Subscriber("/move_base_simple/goal",PoseStamped,callback_goal)
    init_run_pub = rospy.Publisher('initial_run',Bool,queue_size=10)

    cmd = '../ros/move_robot/devel/lib/move_robot/move_robot 1'
    os.system(cmd)

    rospy.spin()


