#include <ros/ros.h>
#include <std_msgs/UInt64.h>
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

#include <donkeycar_msgs/motion_cmd.h>
#include <recorder/recorderConfig.h>

#include <boost/filesystem.hpp>
#include <fstream>
#include <ctime>

class Recorder
{
    public:
    Recorder(ros::NodeHandle& nodeHandle):
    imageSub(nodeHandle, "/camera/image_raw", 10),
    steerSub(nodeHandle, "/combined_steer", 10),
    throttleSub(nodeHandle, "/combined_throttle", 10),
    syncSub(ApproxSyncPolicy(10), imageSub, steerSub, throttleSub)
    {
        std::string nodeName = ros::this_node::getName();
        // boost::filesystem::path dataFolder = nodeHandle.param<std::string>(nodeName + "/data_folder", "data");
        
        imageCount = 0;
        savedImageCount = 0;
        lastSavedImageCount = 0;
        enableFromCfg = false;
        enableFromJs = false;
        downsampleRate = 1;
        recordButton = 5;
        recordButtonState = 0;
        compressOnExit = false;

        syncSub.registerCallback(boost::bind(&Recorder::syncCallback,this,boost::placeholders::_1,boost::placeholders::_2, boost::placeholders::_3));

        joySub = nodeHandle.subscribe<sensor_msgs::Joy>("joy", 10, boost::bind(&Recorder::joyCallback, this, boost::placeholders::_1));
        joyFeedbackPub = nodeHandle.advertise<sensor_msgs::JoyFeedbackArray>("/joy/set_feedback", 10);
        server.setCallback(boost::bind(&Recorder::serverCallback,this,boost::placeholders::_1,boost::placeholders::_2));

        savedCountPub = nodeHandle.advertise<std_msgs::UInt64>("/recorder/saved_count", 10);
    }

    ~Recorder()
    {
        boost::filesystem::path dataFolder = imageFolder.parent_path();
        if(boost::filesystem::exists(dataFolder))
        {
            labelFile.close();
            compressDataFile(dataFolder);
        }
    }

    using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                                                            donkeycar_msgs::motion_cmd,
                                                                            donkeycar_msgs::motion_cmd>;

    private:

    dynamic_reconfigure::Server<recorder::recorderConfig> server;

    message_filters::Subscriber<sensor_msgs::Image> imageSub;
    message_filters::Subscriber<donkeycar_msgs::motion_cmd> steerSub;
    message_filters::Subscriber<donkeycar_msgs::motion_cmd> throttleSub;
    message_filters::Synchronizer<ApproxSyncPolicy> syncSub;
    ros::Subscriber joySub;
    ros::Publisher joyFeedbackPub;
    ros::Publisher savedCountPub;

    boost::filesystem::path imageFolder;
    boost::filesystem::path labelFilePath;
    std::ofstream labelFile;

    uint64_t imageCount;
    uint64_t savedImageCount;
    uint64_t lastSavedImageCount;

    bool enableFromJs;
    bool enableFromCfg;

    int recordButton;
    int recordButtonState;
    int downsampleRate;

    bool dataFolderCreated;
    bool compressOnExit;

    void joyCallback(const sensor_msgs::Joy::ConstPtr& msg)
    {
        try
        {
            if(msg->buttons[recordButton])
            {
                if(msg->buttons[recordButton] != recordButtonState)
                    ROS_INFO("Start recording");
                
                enableFromJs = true;

                // sensor_msgs::JoyFeedbackArray feedbackArray;
                // sensor_msgs::JoyFeedback feedback;
                // feedback.type = sensor_msgs::JoyFeedback::TYPE_RUMBLE;  // 0: RUMBLE feedback
                // feedback.id = 0;                                       // Feedback device ID
                // feedback.intensity = 1.0;   
                // feedbackArray.array.push_back(feedback);  
                // joyFeedbackPub.publish(feedbackArray);
            }
            else
            {
                if(msg->buttons[recordButton] != recordButtonState)
                {
                    ROS_INFO("Stop recording, saved %ld images, there are %ld images in total", savedImageCount - lastSavedImageCount, savedImageCount);
                    lastSavedImageCount = savedImageCount;
                }

                enableFromJs = false;
                
            }

            recordButtonState = msg->buttons[recordButton];
        }
        catch(std::exception& e)
        {
            ROS_ERROR_THROTTLE(1.0, "Invaild button number: %s", e.what());
        }
    }


    void syncCallback(const sensor_msgs::ImageConstPtr& image, 
        const boost::shared_ptr<const donkeycar_msgs::motion_cmd>& steerMsg,
        const boost::shared_ptr<const donkeycar_msgs::motion_cmd>& throttleMsg
    )
    {
        bool enable = enableFromCfg || enableFromJs;

        if(!enable)
            return;

        if(imageCount % downsampleRate == 0)
        {

            cv_bridge::CvImageConstPtr cvImage;
            try
            {
                cvImage = cv_bridge::toCvShare(image, image->encoding);
            }
            catch(cv_bridge::Exception& e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
            }

            savedImageCount += 1;
            if(!boost::filesystem::exists(imageFolder))
            {
                boost::filesystem::create_directories(imageFolder);
                ROS_INFO_STREAM("Created folder " + boost::filesystem::absolute(imageFolder).string());
            }
            if(!boost::filesystem::exists(labelFilePath))
            {
                labelFile = std::ofstream(labelFilePath.string());
                ROS_INFO_STREAM("Opened label file " + labelFilePath.string());
            }

            std::string imageName = std::to_string(savedImageCount) + ".jpg";
            boost::filesystem::path imageFile = imageFolder / imageName;

            cv::imwrite(imageFile.string(), cvImage->image);
            labelFile << imageName << "," << steerMsg->value << "," << throttleMsg->value << std::endl;
            ROS_DEBUG("Image %s saved!", imageFile.string().c_str());
            
            std_msgs::UInt64 msg;
            msg.data = savedImageCount;
            savedCountPub.publish(msg);
        }

        imageCount += 1;
    }

    void serverCallback(recorder::recorderConfig &config, uint32_t level)
    {
        if(level & 0x1)
        {
            enableFromCfg = config.enable;
        }
        if(level & 0x2)
        {
            if(enableFromCfg || enableFromJs)
            {
                ROS_WARN("You can not change the data folder while recording");
            }
            else
            {
                boost::filesystem::path dataFolder = makeFilename(config.data_folder);
                ros::NodeHandle("~").setParam("data_folder_resolved", dataFolder.string());
                if(dataFolder != imageFolder.parent_path())
                {
                    labelFile.close();
                    imageFolder = dataFolder / "images";
                    labelFilePath = imageFolder.parent_path() / "labels.csv";
                    imageCount = 0;
                    savedImageCount = 0;
                }
            }
        }
        if(level & 0x4)
        {
            if(config.downsample_rate <= 0)
                ROS_WARN("Invalid downsampling rate %d", config.downsample_rate);
            else
                downsampleRate = config.downsample_rate;
        }
        if(level & 0x8)
        {
            if(enableFromCfg || enableFromJs)
                ROS_WARN("You can not change the record button while recording");
            else
                recordButton = config.record_button;
        }
        if(level & 0xF)
        {
            compressOnExit = config.compress_on_exit;
        }
    }

    boost::filesystem::path makeFilename(const std::string& pattern)
    {
        std::time_t t = std::time(nullptr);
        std::tm tm = *std::localtime(&t);

        char buffer[256];
        std::strftime(buffer, sizeof(buffer), pattern.c_str(), &tm);

        return boost::filesystem::path(buffer);
    }

    void compressDataFile(const boost::filesystem::path& dataFolder)
    {
        boost::filesystem::path tarFolder = dataFolder.parent_path();
        boost::filesystem::path tarName = dataFolder.filename().replace_extension("tar.gz");\
        boost::filesystem::path tarPath = tarFolder / tarName;

        std::string cmd = "tar -czf '" +  tarPath.string()  + "' -C '" + dataFolder.string() + "' .";
        if(!std::system(cmd.c_str()))
            ROS_ERROR("Failed to compress the data folder");
        else
            ROS_INFO_STREAM("Saved compressed data folder to " + tarPath.string());
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

