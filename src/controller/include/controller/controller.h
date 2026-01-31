#include <ros/ros.h>
#include <donkeycar_msgs/motion_cmd.h>

#include <controller/ControllerConfig.h>
#include <dynamic_reconfigure/server.h>

#include <cmath>

namespace controller{

    class Controller
    {
        public:
        Controller():server(ros::NodeHandle("~controller"))
        {
            ros::NodeHandle nodeHandle("~");
            cmdPub = nodeHandle.advertise<donkeycar_msgs::motion_cmd>("/motion_cmd",10);

            nodeName = ros::this_node::getName();

            throttleForwardRatio = 1.0;
            throttleReverseRatio = 1.0;
            steerRatio = 1.0;
            
            server.setCallback(boost::bind(&Controller::serverCallback,this,boost::placeholders::_1,boost::placeholders::_2));

            ROS_INFO("-----------------------------------------------");
            ROS_INFO(" Controller %s Configuration", nodeName.c_str());
            ROS_INFO("-----------------------------------------------");
            ROS_INFO("%-20s | %-15s ", "Parameter", "Value");
            ROS_INFO("-----------------------------------------------");
            ROS_INFO("%-20s | %-15f ", "Throttle forward ratio", throttleForwardRatio);
            ROS_INFO("%-20s | %-15f ", "Throttle reverse ratio", throttleReverseRatio);
            ROS_INFO("%-20s | %-15f ", "Steer ratio", steerRatio);
            ROS_INFO("-----------------------------------------------");
        }
        void control(float throttle, float steer)
        {
            donkeycar_msgs::motion_cmd cmd;
            cmd.header.stamp = ros::Time::now();
            cmd.header.frame_id = nodeName;
            
            if(throttle > 0)
                cmd.throttle = throttle * throttleForwardRatio;
            else
                cmd.throttle = throttle * throttleReverseRatio;

            cmd.steer = steer * steerRatio;
            cmdPub.publish(cmd);

            ROS_DEBUG("Sent motion cmd: throttle %f steer %f from node %s", cmd.throttle, cmd.steer, nodeName.c_str());
        }

        private:

        ros::Publisher cmdPub;
        ros::Timer timer;
        std::string nodeName;
        float throttleForwardRatio;
        float throttleReverseRatio;
        float steerRatio;

        dynamic_reconfigure::Server<controller::ControllerConfig> server;

        void serverCallback(controller::ControllerConfig &config, uint32_t level)
        {
            if (level & 0x1) 
            {
                ROS_DEBUG("Throttle forward ratio changed: %f", config.throttle_forward_ratio);
                throttleForwardRatio = config.throttle_forward_ratio;
            }
            if (level & 0x2)
            {
                ROS_DEBUG("Throttle reverse ratio changed: %f", config.throttle_reverse_ratio);
                throttleReverseRatio = config.throttle_reverse_ratio;
            }
            if (level & 0x4)
            {
                ROS_DEBUG("Steer ratio changed: %f", config.steer_ratio);
                steerRatio = config.steer_ratio;
            }

        }
    };

}