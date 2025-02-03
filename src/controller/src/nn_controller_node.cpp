#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>

#include <fstream>
#include <algorithm>
#include <numeric>
#include <vector>

#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

#include <controller/controller.h>
#include <inferencer/inferencer.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class NNController: controller::Controller
{
    public:
    NNController(ros::NodeHandle& nodeHandle):nh(nodeHandle),
        status(Status::INIT_BACKEND),
        inferencer(nullptr)
    {
        timer = nodeHandle.createTimer(ros::Duration(1), boost::bind(&NNController::timerCallback, this, boost::placeholders::_1));
    }

    ~NNController()
    {
            
    }

    private:
    enum class Status
    {
        INIT_BACKEND,
        LOAD_MODEL,
        ALLOC_OUTPUT,
        ALLOC_INPUT,
        WAIT_IMAGE,
        RUNNING,
    };

    ros::NodeHandle& nh;
    Status status;
    std::unique_ptr<Inferencer> inferencer;
    cv::Mat imageMat;

    ros::Subscriber imageSub;

    void* outputBuffer;
    void* inputBuffer;
    size_t inputBufferSize;
    size_t outputBufferSize;

    ros::Timer timer;

    cv::Rect roi;

    void timerCallback(const ros::TimerEvent& event)
    {
        if(status == Status::INIT_BACKEND)
        {
            std::string backend = nh.param<std::string>(ros::this_node::getName() + "/backend", "tensorflow");

            try
            {
                inferencer = std::make_unique<Inferencer>(backend);
            }
            catch(const std::runtime_error& e)
            {
                ROS_ERROR_ONCE("Failed to initialize the %s backend: %s. Will retry",backend.c_str(),e.what());
                return;
            }

            ROS_INFO("Successfully initialize %s backend", backend.c_str());
            status = Status::LOAD_MODEL;
        }
        else if(status == Status::LOAD_MODEL)
        {
            std::string modelName;
            if(!nh.getParam("model_file", modelName))
            {
                std::string defaultFolder = ros::package::getPath("donkeycar");
                if(defaultFolder.empty())
                {
                    ROS_FATAL("Cound not find package donkeycar. Exiting");
                    ros::shutdown();
                    return;
                }
                modelName = (boost::filesystem::path(defaultFolder) / "models" / "model").string();
            }

            bool result;

            try
            {
                result = inferencer->loadModel(modelName.c_str());
            }
            catch(const std::runtime_error& e)
            {
                ROS_FATAL("Failed to load model file %s: %s. Existing", modelName.c_str(),e.what());
                ros::shutdown();
                return;
            }

            if(!result)
            {
                ROS_ERROR_ONCE("Failed to load model file %s: %s. Will retry",modelName.c_str(), inferencer->getErrorString());
                return;
            }

            ROS_INFO("Successfully load model file %s", modelName.c_str());
            status = Status::ALLOC_OUTPUT;   
        }
        else if(status == Status::ALLOC_OUTPUT)
        {
            std::string outputName = nh.param<std::string>("output_name", "outputs");

            try
            {
                outputBufferSize = inferencer->getOutputBuffer(outputName.c_str(), &outputBuffer);
            }
            catch(const std::runtime_error& e)
            {
                ROS_FATAL("Failed to allocate ouput tensor: %s. Exiting", e.what());
                ros::shutdown();
                return;
            }

            if(outputBufferSize == 0)
            {
                ROS_ERROR_ONCE("Failed to allocate output tensor: %s. Will retry", inferencer->getErrorString());
                return;
            }

            if(outputBufferSize != sizeof(float))
            {
                ROS_FATAL("Invaild byte size of output tensor. Need 4 but get %ld. Exiting ", outputBufferSize);
                ros::shutdown();
                return;
            }

            ROS_INFO("Successfully allocate output tensor %s", outputName.c_str());
            status = Status::ALLOC_INPUT;

        }
        else if(status == Status::ALLOC_INPUT)
        {
            std::string inputName = nh.param<std::string>("input_name", "input");
            
            try
            {
                inputBufferSize = inferencer->getInputBuffer(inputName.c_str(), &inputBuffer);
            }
            catch(const std::runtime_error& e)
            {
                ROS_FATAL("Failed to allocate input tensor %s: %s. Exi ting", inputName.c_str(), e.what());
                ros::shutdown();
                return;
            }

            if(inputBufferSize == 0)
            {
                ROS_ERROR_ONCE("Failed to allocate input tensor %s: %s. Will retry",inputName.c_str(),inferencer->getErrorString());
                return;
            }

            ROS_INFO("Successfully allocate input tensor %s", inputName.c_str());

            imageSub = nh.subscribe<sensor_msgs::Image>("image_raw", 10, 
                    boost::bind(&NNController::imageCallback1, this, boost::placeholders::_1));

            status = Status::WAIT_IMAGE;
        }
    }

    void imageCallback1(const sensor_msgs::ImageConstPtr& msg) 
    {
        int imageWidth = msg->width;
        int imageHeight = msg->height;

        int x = nh.param<int>("roi/x", 0);
        int y = nh.param<int>("roi/y", 0);
        int roiWidth = nh.param<int>("roi/width", msg->width);
        int roiHeight = nh.param<int>("roi/height", msg->height);

        cv::Rect vaildRegion(0,0,msg->width,msg->height);
        roi = cv::Rect(x,y,roiWidth,roiHeight) & vaildRegion;

        size_t roiByteSize;

        int channels = 0;
        if(msg->encoding == sensor_msgs::image_encodings::MONO8)
        {
            channels = 1;
        }
        else if(msg->encoding == sensor_msgs::image_encodings::RGB8)
        {
           channels = 3;
        }
        else
        {
            ROS_ERROR_ONCE("Unsupported image encoding %s. Will retry", msg->encoding.c_str());
            return;
        }

        roiByteSize = roi.area() * 4 * channels;

        if(roiByteSize != inputBufferSize)
        {
            ROS_ERROR_ONCE("Image roi byte size %ld does not match the input buffer size %ld. Will retry", roiByteSize, inputBufferSize);
            return;
        }

        ROS_INFO("Vaild roi region is x: %d, y: %d, width: %d, height: %d", roi.x, roi.y, roi.width, roi.height);


        imageMat = cv::Mat(roi.height, roi.width, channels==1 ? CV_32FC1 : CV_32FC3, inputBuffer);

        status = Status::RUNNING;

        imageSub = nh.subscribe<sensor_msgs::Image>("image_raw", 10, 
                    boost::bind(&NNController::imageCallback2, this, boost::placeholders::_1));

    }

    void imageCallback2(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImageConstPtr imagePtr = cv_bridge::toCvShare(msg, msg->encoding);

        imagePtr->image(roi).convertTo(imageMat, imagePtr->image.channels()==1 ? CV_32FC1 : CV_32FC3);
        

        if(!inferencer->infer())
        {
            ROS_ERROR("Failed to run inference %s", inferencer->getErrorString());
            imageSub.shutdown();
            status = Status::INIT_BACKEND;
            return;
        }

        this->control(0, *reinterpret_cast<float*>(outputBuffer));

        ROS_DEBUG("Run inference successfully %f", *reinterpret_cast<float*>(outputBuffer));
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "nn_controller_node");
    
    ros::NodeHandle nh("~");

    NNController controller(nh);

    ros::spin();

    return 0;
}






