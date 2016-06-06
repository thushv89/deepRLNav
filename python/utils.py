
#COMMON TOPICS
CAMERA_IMAGE_TOPIC = "/camera/image"
LASER_SCAN_TOPIC = "/obs_scan"
ODOM_TOPIC = "/odom"
CMD_VEL_TOPIC = "/cmd_vel"

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
IMG_W, IMG_H = 640, 480
THUMBNAIL_W, THUMBNAIL_H = 128, 96

#BUMP DETECTION
BUMP_1_THRESH = 0.5
BUMP_02_THRESH = 0.65
NO_RETURN_THRESH = 0.002

#LASER
LASER_FREQUENCY = 20
LASER_POINT_COUNT = 120
LASER_ANGLE = 120

PREF_FREQUENCY = 5

#COMMANDS
GOAL_CANCEL_CMD = "rosservice call /autonomy/path_follower/cancel_request"
MOVE_ROBOT_CMD = "../ros/move_robot/devel/lib/move_robot/move_robot "