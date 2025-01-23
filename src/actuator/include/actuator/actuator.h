#include <ros/ros.h>
#include <controller/motion_cmd.h>
#include <controller/controller.h>
#include <array>
#include <numeric>
#include <unordered_map>

namespace actuator{
    class Actuator 
    {
        public:
        Actuator() = delete;
        Actuator(ros::NodeHandle& nodeHandle)
        {
            std::string nodeName = ros::this_node::getName();

            int controlFreq = nodeHandle.param<int>(nodeName + "/control_frequency",50);
            if(controlFreq < 0)
            {
                ROS_WARN("Invaild control frequency %d, Using default value 50", controlFreq);
            }
            
            timer = nodeHandle.createTimer(ros::Duration(1/controlFreq), boost::bind(&Actuator::timerCallback, this, boost::placeholders::_1));

            cmdSub = nodeHandle.subscribe<controller::motion_cmd>("motion_cmd",10, 
                boost::bind(&Actuator::motionCallback, this, boost::placeholders::_1));
            combinedCmdPub = nodeHandle.advertise<controller::motion_cmd>("combined_motion_cmd",10);
        }

        virtual void actuate(float throttle, float steer)
        {
            
        }

        private:
        ros::Subscriber cmdSub;
        ros::Publisher combinedCmdPub;
        ros::Timer timer;

        std::unordered_map<std::string, std::array<float, 2>> motionCmds;
        
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
            
            controller::motion_cmd combinedCmd;
            combinedCmd.header.stamp = ros::Time::now();
            combinedCmd.steer = combinedSteer;
            combinedCmd.throttle = combinedThrottle;

            combinedCmdPub.publish(combinedCmd);

            ROS_DEBUG("Combine motion cmd: throttle %f steer %f", combinedThrottle, combinedSteer);
            
            this -> actuate(combinedThrottle, combinedSteer);
        }

        void motionCallback(const boost::shared_ptr<const controller::motion_cmd>& msg)
        {
            ROS_DEBUG("Received motion cmd: throttle %f steer %f from node: %s", msg->throttle, msg->steer, msg->header.frame_id.c_str());
            
            motionCmds[msg->header.frame_id][0] = msg->throttle;
            motionCmds[msg->header.frame_id][1] = msg->steer;
        }
    };
}