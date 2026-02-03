#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <controller/controller.h>

#include <controller/JoystickControllerConfig.h>
#include <dynamic_reconfigure/server.h>

#include <string>
#include <stdexcept>


class JoystickController: controller::Controller
{
    public:
    JoystickController(ros::NodeHandle& nodeHandle):server(nodeHandle)
    {
        throttle_axis = 1;
        steer_axis = 0;
        server.setCallback(boost::bind(&JoystickController::serverCallback,this,boost::placeholders::_1,boost::placeholders::_2));

        ROS_INFO("Joystick axis %d is mapped to throttle, axis %d is mapped to steer", throttle_axis, steer_axis);

        joystickSub = nodeHandle.subscribe<sensor_msgs::Joy>("/joy", 10, boost::bind(&JoystickController::callback,this,boost::placeholders::_1));
    }

    private:
    
    int throttle_axis;
    int steer_axis;
    ros::Subscriber joystickSub;

    dynamic_reconfigure::Server<controller::JoystickControllerConfig> server;

    void callback(const sensor_msgs::Joy::ConstPtr& msg)
    {   
        float throttle = 0;
        try
        {
            throttle = msg->axes.at(throttle_axis);
        }
        catch(const std::out_of_range& e)
        {
            ROS_WARN_ONCE("Invaild joystick axis number %d for throttle.", throttle_axis);
        }

        // if(throttle > 0)
        //     throttle = std::log(1 + throttle * 1.71828);
        // else
        //     throttle = -std::log(1 - throttle * 1.71828);

        float steer = 0;
        try
        {
            steer = msg->axes.at(steer_axis);
        }
        catch(const std::out_of_range& e)
        {
            ROS_WARN_ONCE("Invaild joystick axis number %d for steer.", steer_axis);
        }

        controlThrottle(throttle);
        controlSteer(steer);
    }

    void serverCallback(controller::JoystickControllerConfig &config, uint32_t level)
    {
        if (level & 0x1) 
        {
            try
            {
                throttle_axis = std::stoi(config.throttle_axis);
            }
            catch(const std::exception& e)
            {
                ROS_WARN("Invaild joystick axis parameter %s for throttle", config.throttle_axis.c_str());
                return;
            }
            ROS_DEBUG("Parameter 'throttle_axis' changed: %s", config.throttle_axis.c_str());
        }
        if (level & 0x2)
        {
            try
            {
                steer_axis = std::stoi(config.steer_axis);
            }
            catch(const std::exception& e)
            {
                ROS_WARN("Invaild joystick axis parameter %s for steer", config.steer_axis.c_str());
                return;
            }
            ROS_DEBUG("Parameter 'steer_axis' changed: %s", config.steer_axis.c_str());
        }

    }

};

int main(int argc, char** argv)
{
    // Initialize the ROS node
    ros::init(argc, argv, "joystick_controller_node");

    // Create a NodeHandle
    ros::NodeHandle nh("~");


    // Create the controller
    JoystickController controller(nh);

    // Spin to keep the node running and processing callbacks
    ros::spin();

    return 0;
}