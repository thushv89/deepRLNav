#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include "std_msgs/String.h"
#include "std_msgs/Bool.h"
#include <sstream>
using namespace std;

void obsCallback(const std_msgs::Bool::ConstPtr& msg)
{
    ros::NodeHandle n;
    ros::Rate loop_rate(1);
    tf::TransformListener listener;
    tf::StampedTransform mytransform;
    try{
	    ros::Duration(0.5).sleep();
	    listener.waitForTransform("map","base_link",ros::Time::now(),ros::Duration(3.0));
	    listener.lookupTransform("map", "base_link",
				  ros::Time(0), mytransform);
	  
	}
	catch (tf::TransformException &ex) {
	    ROS_ERROR("%s",ex.what());
	    ros::Duration(1.0).sleep();
	}

    double x = double(mytransform.getOrigin().getX());
	double y = double(mytransform.getOrigin().getY());

    cout << "Hit obstacle. Staying at " << x << "," << y << " location";
    ros::Publisher goal_pub = n.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1000);

    geometry_msgs::PoseStamped goal;
	goal.header.frame_id = "map";
	goal.header.stamp = ros::Time::now();
	
	goal.pose.position.x = x;
	goal.pose.position.y = y;

	goal_pub.publish(goal);

    ros::spinOnce();
	loop_rate.sleep();
	
}

int main(int argc, char* argv[]){
    ros::init(argc, argv, "simple_navigation_goals_cancel");

    ros::NodeHandle n;    
    ros::Subscriber sub = n.subscribe("/obstacle_status", 1000, obsCallback);

    ros::spin();
    return 0;
}
