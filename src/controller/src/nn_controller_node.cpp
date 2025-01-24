#include <ros/ros.h>
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
    NNController() = delete;
    NNController(ros::NodeHandle& nodeHandle):controller::Controller(nodeHandle), steerAngle(nullptr)
    {

        std::string nodeName = ros::this_node::getName();

        boost::filesystem::path modelFile = nodeHandle.param<std::string>(nodeName + "/model_file", (boost::filesystem::current_path() / "model" / "model.onnx").string());
        std::string inputName = nodeHandle.param<std::string>(nodeName + "/input_name", "input");
        std::string outputName = nodeHandle.param<std::string>(nodeName + "/output_name", "outputs");
        grayScale = nodeHandle.param<bool>(nodeName + "/grayscale", true);
        imageWidth = nodeHandle.param<int>(nodeName + "/image_width", 1280);
        imageHeight = nodeHandle.param<int>(nodeName + "/image_height", 720);

        if(imageWidth <= 0)
        {
            ROS_WARN_ONCE("Invaild imageWidth. Using default value 1280");
        }

        if(imageHeight <= 0)
        {
            ROS_WARN_ONCE("Invaild imageHeight. Using defaule value 720");
        }

        ROS_INFO_ONCE("Setting up tensorrt environment for model %s, this may take a while, please be patient", modelFile.c_str());
        infer = std::make_unique<inferencer::RTInferencer>(modelFile);

        imageSub = nodeHandle.subscribe<sensor_msgs::Image>("/camera/image_raw", 10, boost::bind(&NNController::imageCallback, this, boost::placeholders::_1));

        int imageStep = 0;  // Row stride in bytes
        int imageType = 0;
        if(grayScale)
        {
            imageStep = imageWidth * sizeof(float);
            imageType = CV_32FC1;
        }
        else
        {
            imageStep = imageWidth * sizeof(float) * 3;
            imageType = CV_32FC3;
        }

        void* inputBuffer = nullptr;
        size_t inputBufferSize = infer->getInputBuffer(inputName, &inputBuffer);
        if(inputBufferSize == 0)
        {
            throw std::runtime_error("Failed to allocate the input tensor: " + infer->getErrorString());
        }
        else if(inputBufferSize != imageStep * imageHeight)
        {
            throw std::runtime_error("Invaild input tensor size");
        }

        imageMat =  cv::Mat(imageHeight, imageWidth, imageType, inputBuffer, imageStep);


        size_t outputBufferSize = infer->getOutputBuffer(outputName, &steerAngle);

        if(outputBufferSize == 0)
        {
            throw std::runtime_error("Failed to allocate the output tensor: " + infer->getErrorString());
        }
        else if(outputBufferSize != sizeof(float))
        {
            throw std::runtime_error("Invaild output tensor size");
        }

        ROS_INFO("-----------------------------------------------");
        ROS_INFO(" NN Configuration");
        ROS_INFO("-----------------------------------------------");
        ROS_INFO("%-20s | %-10s ", "Parameter", "Value");
        ROS_INFO("---------------------+------------+------------");
        ROS_INFO("%-20s | %-10d ", "Image width", imageWidth);
        ROS_INFO("%-20s | %-10d ", "Image height", imageHeight);
        ROS_INFO("%-20s | %-10d ", "Gray scale", grayScale);
        ROS_INFO("-----------------------------------------------");
    }

    ~NNController()
    {
        
    }

    private:

    

    std::unique_ptr<inferencer::RTInferencer> infer;
    ros::Subscriber imageSub;
    cv::Mat imageMat;
    void* steerAngle;
    bool grayScale;
    int imageWidth;
    int imageHeight;


    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {

        if(msg->height != imageHeight || msg->width != imageWidth)
        {
            ROS_WARN_ONCE("Image size does not match. Image size of the ros message is %d*%d while the input image size of the neural network is %d*%d", 
                msg->width, msg->height, imageWidth, imageHeight);
            return;
        }

        cv_bridge::CvImageConstPtr imagePtr = cv_bridge::toCvShare(msg, msg->encoding);

        if(grayScale)
        {
            cv::Mat grayImage;
            cv::cvtColor(imagePtr->image, grayImage, cv::COLOR_RGB2GRAY);
            grayImage.convertTo(imageMat, CV_32FC1);
        }
        else
        {
            imagePtr->image.convertTo(imageMat, CV_32FC1);
        }
        

        if(!infer->infer())
            throw std::runtime_error("Failed to run inference");

        this->control(0, *static_cast<float*>(steerAngle));
    }

};

enum class NNControllerStatus
{
    INITING,
    RUNNING,
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "nn_controller_node");
    
    ros::NodeHandle nh;

    NNControllerStatus status = NNControllerStatus::INITING;

    std::unique_ptr<NNController> controllerPtr;

    while(ros::ok())
    {
        if(status == NNControllerStatus::INITING)
        {
            try
            {
                controllerPtr.reset();
                controllerPtr = std::make_unique<NNController>(nh);
            }
            catch(const std::runtime_error& e)
            {
                ROS_ERROR_ONCE("%s, will retry", e.what());
                boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
                continue;
            }
            status = NNControllerStatus::RUNNING;
        }
        else if(status == NNControllerStatus::RUNNING)
        {
            try
            {
                ros::spin();
            }
            catch(const std::runtime_error& e)
            {
                ROS_ERROR_ONCE("%s, will retry", e.what());
                status = NNControllerStatus::INITING;
                continue;
            }
            break;
        }
    }

    return 0;
}






