

############# THE MOST IMPORTANT SETTING ################

TYPE = 'REAL' #SIM or REAL

#########################################################

#COMMON TOPICS
CAMERA_IMAGE_TOPIC = "/camera/image_color" if TYPE == 'REAL' else "/camera/image/" # sim /camera/image real /camera/image_color
LASER_SCAN_TOPIC = "/scan" # sim and real /scan
ODOM_TOPIC = "/odom"
CMD_VEL_TOPIC = "/wombot/cmd_vel" if TYPE == 'REAL' else "/cmd_vel" #sim /cmd_vel real /wombot/cmd_vel

#MY TOPICS
ACTION_STATUS_TOPIC = "/action_status"
DATA_SENT_STATUS = "/data_sent_status"
DATA_INPUT_TOPIC = "/data_inputs"
DATA_LABEL_TOPIC = "/data_labels"
OBSTACLE_STATUS_TOPIC = "/obstacle_status"
RESTORE_AFTER_BUMP_TOPIC = "/restored_bump"
INITIAL_RUN_TOPIC = "/initial_run"
GOAL_TOPIC = "/move_base_simple/goal"

# CAMERA
CAMERA_FREQUENCY = 30 if TYPE=='REAL' else 10
IMG_W, IMG_H = 640, 480
THUMBNAIL_W, THUMBNAIL_H = 128, 96
IMG_SAVE_SKIP = 5 # how many images skipped when saving sequence

#BUMP DETECTION
BUMP_1_THRESH = 0.25 if TYPE=='REAL' else 0.6 #sim 0.6 real 0.2
BUMP_02_THRESH = 0.18 if TYPE=='REAL' else 0.7 #sim 0.7 real 0.25
NO_RETURN_THRESH = 0.05
REVERSE_PUBLISH_DELAY = 0.15 # real 0.12
ZERO_VEL_PUBLISH_DELAY = 0.01 # publish 0 valued cmd_vel data

#LASER
LASER_FREQUENCY = 40 if TYPE=='REAL' else 10 #sim 10 real 40 
LASER_POINT_COUNT = 1080 if TYPE=='REAL' else 180
LASER_ANGLE = 270 if TYPE=='REAL' else 180 # 
PREF_FREQUENCY = 30 if TYPE=='REAL' else 10

#COMMANDS
GOAL_CANCEL_CMD = "rosservice call /autonomy/path_follower/cancel_request"
MOVE_ROBOT_CMD = "./wombot_small_move_robot_v2 " if TYPE=='REAL' else "./atrv_move_robot "# sim ./atrv_move_robot real ./wombot_move_robot
