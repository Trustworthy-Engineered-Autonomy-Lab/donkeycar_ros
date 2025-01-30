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

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <dlfcn.h>

class NNController: controller::Controller
{
    public:
    NNController(ros::NodeHandle& nodeHandle):nh(nodeHandle),
        status(Status::INIT_BACKEND),
        inferencer(nullptr),
        dllHandle(nullptr)
    {
        timer = nodeHandle.createTimer(ros::Duration(1), boost::bind(&NNController::timerCallback, this, boost::placeholders::_1));
    }

    ~NNController()
    {
        if(dllHandle != nullptr)
        {
            if(inferencer != nullptr)
            {
                deleteInferencer(inferencer);
            }
            dlclose(dllHandle);
        }
            
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

    using createInferencerFunc = void*(*)(void*);
    using deleteInferencerFunc = void (*) (void*);
    using loadModelFunc = bool(*)(void*, const char*);
    using getInputBufferFunc = unsigned (*)(void* , const char* , void** );
    using getOutputBufferFunc = unsigned (*)(void* inferencer, const char* outputName, void** buffer);
    using inferFunc = bool(*)(void* inferencer);
    using getErrorStringFunc = const char* (*) (void*);

    createInferencerFunc createInferencer;
    deleteInferencerFunc deleteInferencer;
    loadModelFunc loadModel;
    getInputBufferFunc getInputBuffer;
    getOutputBufferFunc getOutputBuffer;
    inferFunc infer;
    getErrorStringFunc getErrorString;

    ros::NodeHandle& nh;
    Status status;
    void* inferencer;
    cv::Mat imageMat;

    ros::Subscriber imageSub;

    void* outputBuffer;
    void* inputBuffer;
    size_t inputBufferSize;
    size_t outputBufferSize;

    ros::Timer timer;

    cv::Rect roi;
    bool grayScale;

    void* dllHandle;

    void timerCallback(const ros::TimerEvent& event)
    {
        if(status == Status::INIT_BACKEND)
        {
            std::string backend = nh.param<std::string>(ros::this_node::getName() + "/backend", "tensorflow");
            std::string dllName = "lib" + backend + "_inferencer.so";

            dllHandle = dlopen(dllName.c_str(), RTLD_LAZY); 

            if(dllHandle == nullptr)
            {
                ROS_ERROR_ONCE("Unsupported backend %s: %s. Will retry", backend.c_str(), dlerror());
                return;
            }

            createInferencer = (createInferencerFunc)dlsym(dllHandle,"createInferencer");
            deleteInferencer = (deleteInferencerFunc)dlsym(dllHandle,"deleteInferencer");
            loadModel = (loadModelFunc)dlsym(dllHandle,"loadModel");
            getInputBuffer = (getInputBufferFunc)dlsym(dllHandle,"getInputBuffer");
            getOutputBuffer = (getOutputBufferFunc)dlsym(dllHandle,"getOutputBuffer");
            infer = (inferFunc)dlsym(dllHandle,"infer");
            getErrorString = (getErrorStringFunc)dlsym(dllHandle,"getErrorString");

            if (!createInferencer || !deleteInferencer || !loadModel || 
                !getInputBuffer || !getOutputBuffer || !infer || !getErrorString)
            {
                ROS_FATAL("Failed to load function symbols from library file: %s", dlerror());
                ros::shutdown();
                return;
            }

            try
            {
                inferencer = createInferencer(&nh);
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
                result = loadModel(inferencer, modelName.c_str());
            }
            catch(const std::runtime_error& e)
            {
                ROS_FATAL("Failed to load model file %s: %s. Existing", modelName.c_str(),e.what());
                ros::shutdown();
                return;
            }

            if(!result)
            {
                ROS_ERROR_ONCE("Failed to load model file %s: %s. Will retry",modelName.c_str(),getErrorString(inferencer));
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
                outputBufferSize = getOutputBuffer(inferencer, outputName.c_str(), &outputBuffer);
            }
            catch(const std::runtime_error& e)
            {
                ROS_FATAL("Failed to allocate ouput tensor: %s. Exiting", e.what());
                ros::shutdown();
                return;
            }

            if(outputBufferSize == 0)
            {
                ROS_ERROR_ONCE("Failed to allocate output tensor: %s. Will retry", getErrorString(inferencer));
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
                inputBufferSize = getInputBuffer(inferencer, inputName.c_str(), &inputBuffer);
            }
            catch(const std::runtime_error& e)
            {
                ROS_FATAL("Failed to allocate input tensor %s: %s. Exi ting", inputName.c_str(), e.what());
                ros::shutdown();
                return;
            }

            if(inputBufferSize == 0)
            {
                ROS_ERROR_ONCE("Failed to allocate input tensor %s: %s. Will retry",inputName.c_str(),getErrorString(inferencer));
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

        grayScale = nh.param<bool>("to_grayscale", false);

        if(grayScale)
        {
            roiByteSize = roi.area() * 4;
        }
        else
        {
            roiByteSize = roi.area() * 3 * 4;
        }

        if(roiByteSize != inputBufferSize)
        {
            ROS_ERROR_ONCE("Image roi byte size %ld does not match the input buffer size %ld. Will retry", roiByteSize, inputBufferSize);
            return;
        }

        ROS_INFO("Vaild roi region is x: %d, y: %d, width: %d, height: %d", roi.x, roi.y, roi.width, roi.height);

        imageMat = cv::Mat(roi.height, roi.width, CV_32FC1, inputBuffer);

        status = Status::RUNNING;

        imageSub = nh.subscribe<sensor_msgs::Image>("image_raw", 10, 
                    boost::bind(&NNController::imageCallback2, this, boost::placeholders::_1));

    }

    void imageCallback2(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImageConstPtr imagePtr = cv_bridge::toCvShare(msg, msg->encoding);

        if(grayScale)
        {
            cv::Mat grayImage;
            cv::cvtColor(imagePtr->image(roi), grayImage, cv::COLOR_RGB2GRAY);
            grayImage.convertTo(imageMat, CV_32FC1);
        }
        else
        {
            imagePtr->image.convertTo(imageMat, CV_32FC1);
        }
        

        if(!infer(inferencer))
        {
            ROS_ERROR("Failed to run inference %s", getErrorString(inferencer));
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






