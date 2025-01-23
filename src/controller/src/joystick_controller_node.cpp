#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <controller/controller.h>

class JoystickController: controller::Controller
{
    public:
    JoystickController(ros::NodeHandle& nodeHandle):controller::Controller(nodeHandle)
    {
        throttle_axis = nodeHandle.param<int>(ros::this_node::getName()+"/throttle_axis", 4);
        steer_axis = nodeHandle.param<int>(ros::this_node::getName()+"/steer_axis", 0);
        ROS_INFO("Joystick axis %d is mapped to throttle, axis %d is mapped to steer", throttle_axis, steer_axis);

        joystickSub = nodeHandle.subscribe<sensor_msgs::Joy>("/joy", 10, boost::bind(&JoystickController::callback,this,boost::placeholders::_1));
    }

    private:
    
    int throttle_axis;
    int steer_axis;
    ros::Subscriber joystickSub;

    void callback(const sensor_msgs::Joy::ConstPtr& msg)
    {   
        float throttle = 4;
        try
        {
            throttle = msg->axes.at(throttle_axis);
        }
        catch(const std::out_of_range& e)
        {
            ROS_WARN_ONCE("Invaild joystick axis number %d for throttle. Using default value 4", throttle_axis);
        }

        float steer = 0;
        try
        {
            steer = msg->axes.at(steer_axis);
        }
        catch(const std::out_of_range& e)
        {
            ROS_WARN_ONCE("Invaild joystick axis number %d for steer. Using defaule value 0", throttle_axis);
        }

        control(throttle, steer);
    }

};

int main(int argc, char** argv)
{
    // Initialize the ROS node
    ros::init(argc, argv, "joystick_controller_node");

    // Create a NodeHandle
    ros::NodeHandle nh;


    // Create the controller
    JoystickController controller(nh);

    // Spin to keep the node running and processing callbacks
    ros::spin();

    return 0;
}