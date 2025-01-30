#include <ros/ros.h>
#include <controller/motion_cmd.h>

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
            cmdPub = nodeHandle.advertise<controller::motion_cmd>("/motion_cmd",10);

            nodeName = ros::this_node::getName();
            throttleRatio = nodeHandle.param<float>("throttle_ratio", 1);
            steerRatio = nodeHandle.param<float>("steer_ratio", 1);

            ROS_INFO("-----------------------------------------------");
            ROS_INFO(" Controller %s Configuration", nodeName.c_str());
            ROS_INFO("-----------------------------------------------");
            ROS_INFO("%-20s | %-10s ", "Parameter", "Value");
            ROS_INFO("-----------------------------------------------");
            ROS_INFO("%-20s | %-10f ", "Throttle ratio", throttleRatio);
            ROS_INFO("%-20s | %-10f ", "Steer ratio", steerRatio);
            ROS_INFO("-----------------------------------------------");

            server.setCallback(boost::bind(&Controller::serverCallback,this,boost::placeholders::_1,boost::placeholders::_2));
        }
        void control(float throttle, float steer)
        {
            controller::motion_cmd cmd;
            cmd.header.stamp = ros::Time::now();
            cmd.header.frame_id = nodeName;

            if(throttle > 0)
                cmd.throttle = throttleRatio * std::log(1 + throttle * 1.71828);
            else
                cmd.throttle = -throttleRatio * std::log(1 - throttle * 1.71828);

            cmd.steer = steer * steerRatio;
            cmdPub.publish(cmd);

            ROS_DEBUG("Sent motion cmd: throttle %f steer %f from node %s", cmd.throttle, cmd.steer, nodeName.c_str());
        }

        private:

        ros::Publisher cmdPub;
        ros::Timer timer;
        std::string nodeName;
        float throttleRatio;
        float steerRatio;

        dynamic_reconfigure::Server<controller::ControllerConfig> server;

        void serverCallback(controller::ControllerConfig &config, uint32_t level)
        {
            if (level & 0x1) 
            {
                ROS_DEBUG("Parameter 'throttle_ratio' changed: %f", config.throttle_ratio);
                throttleRatio = config.throttle_ratio;
            }
            else if (level & 0x2)
            {
                ROS_DEBUG("Parameter 'steer_ratio' changed: %f", config.steer_ratio);
                steerRatio = config.steer_ratio;
            }

        }
    };

}