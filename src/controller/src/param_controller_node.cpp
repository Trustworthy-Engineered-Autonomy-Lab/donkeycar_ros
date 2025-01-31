#include <controller/controller.h>
#include <controller/ParamControllerConfig.h>

#include <dynamic_reconfigure/server.h>


class ParamController: controller::Controller
{
    public:
    ParamController(ros::NodeHandle& nodeHandle):server(nodeHandle)
    {
        server.setCallback(boost::bind(&ParamController::serverCallback,this,boost::placeholders::_1,boost::placeholders::_2));
    }

    ~ParamController()
    {

    }

    private:

    dynamic_reconfigure::Server<controller::ParamControllerConfig> server;
    float throttle;
    float steer;

    void serverCallback(controller::ParamControllerConfig &config, uint32_t level)
    {
        if(level & 0x1)
        {
            throttle = config.throttle_value;
        }
        if(level & 0x2)
        {
            steer = config.steer_angle;
        }
        this->control(throttle, steer);
    }
};


int main(int argc, char** argv)
{
    // Initialize the ROS node
    ros::init(argc, argv, "param_controller_node");

    // Create a NodeHandle
    ros::NodeHandle nh("~");


    // Create the controller
    ParamController controller(nh);

    // Spin to keep the node running and processing callbacks
    ros::spin();

    return 0;
}