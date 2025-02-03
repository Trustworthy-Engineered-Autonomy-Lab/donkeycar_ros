#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Joy.h>
#include <sensor_msgs/JoyFeedbackArray.h>
#include <sensor_msgs/JoyFeedback.h>
#include <dynamic_reconfigure/server.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <controller/motion_cmd.h>
#include <recorder/recorderConfig.h>

#include <boost/filesystem.hpp>
#include <fstream>

class Recorder
{
    public:
    Recorder(ros::NodeHandle& nodeHandle):
    imageSub(nodeHandle, "/camera/image_raw", 10),
    motionSub(nodeHandle, "/combined_motion_cmd", 10),
    syncSub(ApproxSyncPolicy(10), imageSub, motionSub),
    imageCount(0)
    {
        std::string nodeName = ros::this_node::getName();
        boost::filesystem::path dataFolder = nodeHandle.param<std::string>(nodeName + "/data_folder", "data");
        
        startButton = nodeHandle.param<int>(nodeName + "/start_button", 7);
        stopButton = nodeHandle.param<int>(nodeName + "/stop_button", 6);
        throttleThreshold = nodeHandle.param<float>(nodeName + "/throttle_threshold", 0.05);
        if(throttleThreshold < 0)
        {
            ROS_WARN("Invaild throttle threshold %f. Using default value 0.05", throttleThreshold);
            throttleThreshold = 0.05;
        }

        openDataFolder(dataFolder);

        syncSub.registerCallback(boost::bind(&Recorder::syncCallback,this,boost::placeholders::_1,boost::placeholders::_2));

        joySub = nodeHandle.subscribe<sensor_msgs::Joy>("joy", 10, boost::bind(&Recorder::joyCallback, this, boost::placeholders::_1));
        joyFeedbackPub = nodeHandle.advertise<sensor_msgs::JoyFeedbackArray>("/joy/set_feedback", 10);
        server.setCallback(boost::bind(&Recorder::serverCallback,this,boost::placeholders::_1,boost::placeholders::_2));
    }

    ~Recorder()
    {

    }

    using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, controller::motion_cmd>;

    private:

    dynamic_reconfigure::Server<recorder::recorderConfig> server;

    message_filters::Subscriber<sensor_msgs::Image> imageSub;
    message_filters::Subscriber<controller::motion_cmd> motionSub;
    message_filters::Synchronizer<ApproxSyncPolicy> syncSub;
    ros::Subscriber joySub;
    ros::Publisher joyFeedbackPub;

    boost::filesystem::path imageFolder;
    std::ofstream labelFile;

    unsigned imageCount;

    bool enable;
    float throttleThreshold;
    float recordOnThrottle;

    int startButton;
    int stopButton;

    void openDataFolder(const boost::filesystem::path& dataFolder)
    {
        if(dataFolder == imageFolder.parent_path().parent_path())
            return;
        
        auto now = std::chrono::system_clock::now();
        std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
        std::tm localTime = *std::localtime(&nowTime);
        std::ostringstream runFolderName;
        runFolderName << "collect_" << std::put_time(&localTime, "%Y-%m-%d_%H-%M-%S");

        boost::filesystem::path runFolder = dataFolder/ runFolderName.str();

        imageFolder = runFolder / "images";
        boost::filesystem::create_directories(imageFolder);

        ROS_INFO_STREAM("Create folder " + boost::filesystem::absolute(runFolder).string());

        labelFile = std::ofstream((runFolder/ "labels.csv").string());
    }

    void closeDataFolder()
    {
        labelFile.close();
    }

    void joyCallback(const sensor_msgs::Joy::ConstPtr& msg)
    {
        try
        {
            if(msg->buttons[startButton])
            {
                ROS_INFO("Start Recording");
                enable = true;

                // sensor_msgs::JoyFeedbackArray feedbackArray;
                // sensor_msgs::JoyFeedback feedback;
                // feedback.type = sensor_msgs::JoyFeedback::TYPE_RUMBLE;  // 0: RUMBLE feedback
                // feedback.id = 0;                                       // Feedback device ID
                // feedback.intensity = 1.0;   
                // feedbackArray.array.push_back(feedback);  
                // joyFeedbackPub.publish(feedbackArray);
            }
            else if(msg->buttons[stopButton])
            {
                ROS_INFO("Stop Recording");
                enable = false;

                
            }
        }
        catch(std::exception& e)
        {
            ROS_ERROR_ONCE("Invaild button number: %s", e.what());
        }
    }


    void syncCallback(const sensor_msgs::ImageConstPtr& image, const boost::shared_ptr<const controller::motion_cmd>& motion)
    {
        if(!enable)
            return;

        if(recordOnThrottle && std::fabs(motion->throttle) < throttleThreshold)
            return;

        cv_bridge::CvImageConstPtr cvImage;
        try
        {
            cvImage = cv_bridge::toCvShare(image, image->encoding);
        }
        catch(cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }

        std::string imageName = std::to_string(imageCount) + ".jpg";
        boost::filesystem::path imageFile = imageFolder/ imageName;

        cv::imwrite(imageFile.string(), cvImage->image);
        labelFile << imageName << "," << motion->steer << "," << motion->throttle << std::endl;

        imageCount += 1;

        ROS_DEBUG("Image %s saved!", imageFile.string().c_str());
    }

    void serverCallback(recorder::recorderConfig &config, uint32_t level)
    {
        if(level & 0x1)
        {
            recordOnThrottle = config.record_on_throttle;
        }
        if(level & 0x2)
        {
            throttleThreshold = config.record_threshold;
        }
        if(level & 0x4)
        {
            enable = config.enable;
        }
        if(level & 0x8)
        {
            if(enable)
            {
                ROS_WARN("You can not change the data folder while recording");
            }
            else
            {
                closeDataFolder();
                openDataFolder(config.data_folder);
            }
        }
    }
};


int main(int argc, char** argv)
{
    // Initialize the ROS node
    ros::init(argc, argv, "recorder_node");

    // Create a NodeHandle
    ros::NodeHandle nh;

    Recorder recoder(nh);

    // Spin to keep the node running and processing callbacks
    ros::spin();

    return 0;
}

