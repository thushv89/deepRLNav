#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include "std_msgs/String.h"
#include "std_msgs/Bool.h"
#include <sstream>
#include <math.h>

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
    tf::Quaternion newQuat;
	cout << "Current location" << x << "," << y << "\n";
	cout << "Current Yaw: " << yaw_angle << "\n";
	double newX = 0;
	double newY = 0;
	
	double stepSize = 0.4; //amount forward
	double stepSize2 = 0.5; //amount sideway
	//double diag = sqrt(pow(stepSize,2)+pow(stepSize2,2));
	double diag = 0.8;
	double theta = 0.0;
	// |------>
	// |
	// V
	if(yaw_angle>=0 && yaw_angle<=1.571){
	    cout << "I am in first quadron\n";
	    theta = yaw_angle;
	    if(action ==2){
		newX = x+(stepSize*cos(theta)+stepSize2*sin(theta));
		newY = y+(stepSize*sin(theta)-stepSize2*cos(theta));
	
	    }else if(action ==1){
		newX = x + diag*cos(theta);
		newY = y + diag*sin(theta);

	    }else if(action == 0){
		newX = x+(stepSize*cos(theta)-stepSize2*sin(theta));
		newY = y+stepSize*sin(theta)+stepSize2*cos(theta);
	    }
	}
	//^
	//|
	//|
	// ---------->
	else if(yaw_angle>=1.571 && yaw_angle<=3.142){
	    cout << "I am in second quadron\n";
	    theta = 3.142-yaw_angle;

	    if(action ==2){
		newX = x-(stepSize*cos(theta)-stepSize2*sin(theta));
		newY = y+(stepSize*sin(theta)+stepSize2*cos(theta));

	    }else if(action ==1){
		newX = x - diag*cos(theta);
		newY = y + diag*sin(theta);

	    }else if(action == 0){
		newX = x-(stepSize*cos(theta)+stepSize2*sin(theta));
		newY = y+(stepSize*sin(theta)-stepSize2*cos(theta));
	    }      
	}
	// 	^
	// 	|
	// <----|    
	else if(yaw_angle>=-3.142 && yaw_angle<=-1.571){
	    cout << "I am in third quadron\n";
	    theta = 3.142 + yaw_angle;
	    if(action ==2){
		newX = x-(stepSize*cos(theta)+stepSize2*sin(theta));
		newY = y-(stepSize*sin(theta)-stepSize2*cos(theta));

	    }else if(action ==1){
		newX = x-diag*cos(theta);
		newY = y-diag*sin(theta);

	    }else if(action == 0){
		newX = x-(stepSize*cos(theta)-stepSize2*sin(theta));
		newY = y-(stepSize*sin(theta)+stepSize2*cos(theta));

	    }
	}
	// <-----
	//	|
	//	|
	//	V
	else if(yaw_angle<=0.0 && yaw_angle>=-1.571){
	  cout << "I am in fourth quadron\n";
	  theta = -yaw_angle;
	  if(action ==2){
	      newX = x+(stepSize*cos(theta)-stepSize2*sin(theta));
	      newY = y-(stepSize*sin(theta)+stepSize2*cos(theta));

	  }else if(action ==1){
	      newX = x + diag*cos(theta);
	      newY = y - diag*sin(theta);

	  }else if(action == 0){
	      newX = x+(stepSize*cos(theta)+stepSize2*sin(theta));
	      newY = y-(stepSize*sin(theta)-stepSize2*cos(theta));
	  }  
	}
	
	if(action == 3){
	    newX = x;
	    newY = y;
	}
	cout << "Absolute angle to X axis is " << theta << "\n";
	cout << "New location" << newX << "," << newY << "\n";

    
	geometry_msgs::PoseStamped goal;
	goal.header.frame_id = "map";
	goal.header.stamp = ros::Time::now();    
    
	goal.pose.position.x = newX;
	goal.pose.position.y = newY;
	//goal.pose.orientation = tf::createQuaternionMsgFromYaw(newYaw);
	
	goal_pub.publish(goal);
	
	ros::spinOnce();
	loop_rate.sleep();
	return 0;
    }

  return 0;
}
