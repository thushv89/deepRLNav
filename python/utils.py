
#COMMON TOPICS
CAMERA_IMAGE_TOPIC = "/camera/image" # sim /camera/image real /camera/image_color
LASER_SCAN_TOPIC = "/scan" # sim /obs_scan real /scan
ODOM_TOPIC = "/odom"
CMD_VEL_TOPIC = "/cmd_vel" #sim /cmd_vel real /wombot/cmd_vel

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
CAMERA_FREQUENCY = 10
IMG_W, IMG_H = 320, 240
THUMBNAIL_W, THUMBNAIL_H = 128, 96
IMG_SAVE_SKIP = 5 # how many images skipped when saving sequence

#BUMP DETECTION
BUMP_1_THRESH = 0.6 #sim 0.5 real 0.03
BUMP_02_THRESH = 0.7 #sim 0.65 real 0.05
NO_RETURN_THRESH = 0.002
REVERSE_PUBLISH_DELAY = 0.3
ZERO_VEL_PUBLISH_DELAY = 0.05 # publish 0 valued cmd_vel data

#LASER
LASER_FREQUENCY = 10
LASER_POINT_COUNT = 180
LASER_ANGLE = 180

PREF_FREQUENCY = 5

#COMMANDS
GOAL_CANCEL_CMD = "rosservice call /autonomy/path_follower/cancel_request"
MOVE_ROBOT_CMD = "./atrv_move_robot "
