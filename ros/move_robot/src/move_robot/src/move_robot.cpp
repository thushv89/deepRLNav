#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include "std_msgs/String.h"
#include <sstream>
using namespace std;


int main(int argc, char* argv[]){


// st is input string
    int action;
    stringstream(argv[1]) >> action;
    cout << "Argument is " << action << "\n";
    ros::init(argc, argv, "simple_navigation_goals");

    ros::NodeHandle n;
    ros::Publisher goal_pub = n.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1000);

    //we'll send a goal to the robot to move 1 meter forward
    ros::Rate loop_rate(1);

    while(n.ok()){
      
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
      
	double yaw_angle = tf::getYaw(mytransform.getRotation()); //yaw will be in rad    
	double x = double(mytransform.getOrigin().getX());
	double y = double(mytransform.getOrigin().getY());

	cout << "Current location" << x << "," << y << "\n";
	cout << "Current Yaw: " << yaw_angle << "\n";
	double newX = 0;
	double newY = 0;
	double newZO = 0;
	double newWO = 0;
	int stepSize = 2;
	int stepSize2 = 1;
	// |
	// |
	// V
	if(yaw_angle<=1.571/2 && yaw_angle>=-1.571/2){
	    cout << "I am facing down\n";
	    if(action ==0){
		newX = x+stepSize2;
		newY = y+stepSize;
		newZO=0.7;
		newWO=0.7;
	    }else if(action ==1){
		newX = x+stepSize;
		newY = y;
		newZO=0;
		newWO=1;
	    }else if(action == 2){
		newX = x+stepSize2;
		newY = y-stepSize;
		newZO=-0.7;
		newWO=0.7;
	    }
	}
	// ---------->
	else if(yaw_angle>=1.571/2 && yaw_angle<=1.571+(1.571/2)){
	    cout << "I am facing right\n";
	    if(action ==0){
		newX = x-stepSize;
		newY = y+stepSize2;
		newZO=1;
		newWO=0;
	    }else if(action ==1){
		newX = x;
		newY = y+stepSize;
		newZO=0.7;
		newWO=0.7;
	    }else if(action == 2){
		newX = x+stepSize;
		newY = y+stepSize2;
		newZO=0;
		newWO=1;
	    }      
	}
	// A
	// |
	// |    
	else if((yaw_angle>=1.571+(1.571/2) && yaw_angle<=3.14) || (yaw_angle<=-1.571-(1.571/2) && yaw_angle>=-3.14) ){
	    cout << "I am facing up\n";
	    if(action ==0){
		newX = x-stepSize2;
		newY = y-stepSize;
		newZO=-0.7;
		newWO=0.7;
	    }else if(action ==1){
		newX = x-stepSize;
		newY = y;
		newZO=1;
		newWO=0;
	    }else if(action == 2){
		newX = x-stepSize2;
		newY = y+stepSize;
		newZO=0.7;
		newWO=0.7;
	    }
	}
	// <---------
	else if(yaw_angle<=0.0 && yaw_angle>=-1.571-(1.571/2) ){
	  cout << "I am facing left\n";
	  if(action ==0){
	      newX = x+stepSize;
	      newY = y-stepSize2;
	      newZO=0;
	      newWO=1;
	  }else if(action ==1){
	      newX = x;
	      newY = y-stepSize;
	      newZO=-0.7;
	      newWO=0.7;
	  }else if(action == 2){
	      newX = x-stepSize;
	      newY = y-stepSize2;
	      newZO=1;
	      newWO=0;
	  }  
	}
	
	if(action == 3){
	  newX = x;
	  newY = y;
	}
	cout << "New location" << newX << "," << newY << "\n";
	
	geometry_msgs::PoseStamped goal;
	goal.header.frame_id = "map";
	goal.header.stamp = ros::Time::now();
	
	goal.pose.position.x = newX;
	goal.pose.position.y = newY;
	goal.pose.orientation.x = 0;
	goal.pose.orientation.y = 0;
	goal.pose.orientation.z = newZO;
	goal.pose.orientation.w = newWO;

	goal_pub.publish(goal);
	
	ros::spinOnce();
	loop_rate.sleep();
	return 0;
    }

  return 0;
}
