
#COMMON TOPICS
CAMERA_IMAGE_TOPIC = "/camera/image_color"
LASER_SCAN_TOPIC = "/scan"
ODOM_TOPIC = "/odom"
CMD_VEL_TOPIC = "/wombot/cmd_vel"

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
CAMERA_FREQUENCY = 30
IMG_W, IMG_H = 640, 480
THUMBNAIL_W, THUMBNAIL_H = 128, 96

#BUMP DETECTION
BUMP_1_THRESH = 0.25 #sim 0.5 real 0.03
BUMP_02_THRESH = 0.30 #sim 0.65 real 0.05
NO_RETURN_THRESH = 0.002
REVERSE_PUBLISH_DELAY = 0.12
ZERO_VEL_PUBLISH_DELAY = 0.05 # publish 0 valued cmd_vel data
#LASER
LASER_FREQUENCY = 40
LASER_POINT_COUNT = 1080
LASER_ANGLE = 270

PREF_FREQUENCY = 5

#COMMANDS
GOAL_CANCEL_CMD = "rosservice call /autonomy/path_follower/cancel_request"
MOVE_ROBOT_CMD = "./wombot_move_robot "
