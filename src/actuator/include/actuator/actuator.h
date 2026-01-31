#include <ros/ros.h>
#include <donkeycar_msgs/motion_cmd.h>
#include <array>
#include <numeric>
#include <unordered_map>

#include <actuator/ActuatorConfig.h>
#include <dynamic_reconfigure/server.h>

namespace actuator{
    class Actuator 
    {
        public:
        Actuator():server(ros::NodeHandle("~actuator"))
        {
            ros::NodeHandle nodeHandle("~");

            nodeName = ros::this_node::getName();
            
            timer = nodeHandle.createTimer(ros::Duration(1.0/20.0), boost::bind(&Actuator::timerCallback, this, boost::placeholders::_1));
            server.setCallback(boost::bind(&Actuator::serverCallback,this,boost::placeholders::_1,boost::placeholders::_2));
            

            steerSub = nodeHandle.subscribe<donkeycar_msgs::motion_cmd>("/steer",10, 
                boost::bind(&Actuator::steerCallback, this, boost::placeholders::_1));
            throttleSub = nodeHandle.subscribe<donkeycar_msgs::motion_cmd>("/throttle",10, 
                boost::bind(&Actuator::throttleCallback, this, boost::placeholders::_1));
            
            combinedSteerPub = nodeHandle.advertise<donkeycar_msgs::motion_cmd>("/combined_steer",10);
            combinedThrottlePub = nodeHandle.advertise<donkeycar_msgs::motion_cmd>("/combined_throttle",10);
        }

        virtual void actuate(float throttle, float steer) = 0;

        private:
        ros::Subscriber steerSub;
        ros::Subscriber throttleSub;
        ros::Publisher combinedSteerPub;
        ros::Publisher combinedThrottlePub;
        ros::Timer timer;
        std::string nodeName;

        std::unordered_map<std::string, std::array<float, 2>> motionCmds;
        dynamic_reconfigure::Server<actuator::ActuatorConfig> server;
        
        void timerCallback(const ros::TimerEvent& event)
        {
            float combinedThrottle = 0;
            float combinedSteer = 0;

            for (const auto& pair : motionCmds)
            {
                combinedThrottle += pair.second[0];
                combinedSteer += pair.second[1];
            } 

            if(combinedThrottle > 1)
                combinedThrottle = 1;
            else if(combinedThrottle < -1)
                combinedThrottle = -1;

            if(combinedSteer > 1)
                combinedSteer = 1;
            else if(combinedSteer < -1)
                combinedSteer = -1;
            
            auto now = ros::Time::now();
            donkeycar_msgs::motion_cmd combinedSteerMsg;
            combinedSteerMsg.header.stamp = now;
            combinedSteerMsg.value = combinedSteer;
            combinedSteerMsg.source = nodeName;
            combinedSteerPub.publish(combinedSteerMsg);

            donkeycar_msgs::motion_cmd combinedThrottleMsg;
            combinedThrottleMsg.header.stamp = now;
            combinedThrottleMsg.value = combinedThrottle;
            combinedThrottleMsg.source = nodeName;
            combinedThrottlePub.publish(combinedThrottleMsg);

            ROS_DEBUG("Sent combined motion cmd: throttle %f steer %f from node %s", combinedThrottle, combinedSteer, nodeName.c_str());
            
            this -> actuate(combinedThrottle, combinedSteer);
        }

        void steerCallback(const boost::shared_ptr<const donkeycar_msgs::motion_cmd>& msg)
        {
            ROS_DEBUG("Received steer cmd %f from node: %s", msg->value, msg->source.c_str());
            motionCmds[msg->source][1] = msg->value;
        }

        void throttleCallback(const boost::shared_ptr<const donkeycar_msgs::motion_cmd>& msg)
        {
            ROS_DEBUG("Received throttle cmd %f from node: %s", msg->value, msg->source.c_str());
            motionCmds[msg->source][0] = msg->value;
        }

        void serverCallback(actuator::ActuatorConfig &config, uint32_t level)
        {
            if (level & 0x1) 
            {
                if(config.control_frequency < 0)
                {
                    ROS_WARN("Invaild control frequency %d", config.control_frequency);
                }
                else
                {
                    ROS_DEBUG("Parameter 'control_frequency' changed: %dHz", config.control_frequency);
                    timer.setPeriod(ros::Duration(1.0/config.control_frequency));
                }
            }

        }
    };
}