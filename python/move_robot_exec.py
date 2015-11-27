import rospy
from std_msgs.msg import Int16
from std_msgs.msg import Bool

def callback_action_status(msg):
    import os
    cmd = '../ros/move_robot/devel/lib/move_robot/move_robot ' + str(msg.data)
    os.system(cmd)

def callback_obstacle_status(msg):
    print("[move_robot_executer] Hit obstacle")
    import os
    cmd = 'rosnode kill /move_robot_executer'    
    os.system(cmd)    
    
  
if __name__=='__main__':  
    
    import os
    cmd = '../ros/move_robot/devel/lib/move_robot/move_robot 1'
    os.system(cmd)
    
    rospy.init_node("move_robot_executer")        
    rospy.Subscriber("/action_status", Int16, callback_action_status)
    rospy.Subscriber("/obstacle_status", Bool, callback_obstacle_status)
    rospy.spin() 