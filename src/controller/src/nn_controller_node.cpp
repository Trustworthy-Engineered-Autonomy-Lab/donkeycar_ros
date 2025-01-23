#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <cuda_runtime_api.h>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include <fstream>
#include <algorithm>
#include <numeric>
#include <vector>

#include <controller/controller.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class Infer
{
    public:
    Infer() = delete;
    Infer(const boost::filesystem::path& filePath):
    buffers{nullptr, nullptr}
    {
        bool needSerialize = false;

        std::string fileExt = filePath.extension().string();

        if(fileExt == ".onnx")
        {
            boost::filesystem::path engineFilePath = filePath;
            engineFilePath.replace_extension("engine");

            if(boost::filesystem::exists(engineFilePath))
            {
                engine = loadEngine(engineFilePath.string());
            }


            if(engine == nullptr)
            {
                engine = loadOnnx(filePath.string());
                if(engine != nullptr)
                    saveEngine(engineFilePath.string(), engine);
            }
        }
        else if(fileExt == ".engine")
        {
            engine = loadEngine(filePath.string());
        }
        else
        {
            throw std::runtime_error("Unsupported model file format " + fileExt);
        }


        if (engine == nullptr)
        {
            throw std::runtime_error("Failed to create the engine " + errorString);
        }

        context.reset(engine->createExecutionContext());

        if(context == nullptr)
        {
            throw std::runtime_error("Failed to create the context");
        }

        cudaError_t error = cudaStreamCreate(&stream);

        if(error != cudaSuccess)
        {
            throw std::runtime_error("Failed to create cuda stream: " + std::string(cudaGetErrorString(error)));
        }

    }

    ~Infer()
    {
        cudaStreamDestroy(stream);
    }

    bool infer()
    {
        if(!context->enqueueV2(buffers.data(), stream, nullptr))
        {
            errorString = "Failed to enqueue the stream";
            return false;
        }

        cudaStreamSynchronize(stream);
        return true;
    }

    bool setInputBuffer(const std::string& inputName, void* buffer, size_t bufferSize)
    {

        int inputIndex = engine->getBindingIndex(inputName.c_str());

        if(inputIndex == -1)
        {
            errorString = "Invaild input index";
            return false;
        }

        nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
        nvinfer1::DataType inputDataType = engine->getBindingDataType(inputIndex);

        size_t inputSize = getVolume(inputDims) * getDataTypeSize(inputDataType);
        
        if(bufferSize != inputSize || buffer == nullptr)
        {
            errorString = "Invaild parameters";
            return false;
        }

        buffers.insert(buffers.begin() + inputIndex ,buffer);

        return true;
    }

    bool setOutputBuffer(const std::string& outputName, void* buffer, size_t bufferSize)
    {
        int outputIndex = engine->getBindingIndex(outputName.c_str());
        if(outputIndex == -1)
        {
            errorString = "Invaild output index";
            return false;
        }

        nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
        nvinfer1::DataType outputDataType = engine->getBindingDataType(outputIndex); 

        size_t outputSize = getVolume(outputDims) * getDataTypeSize(outputDataType);

        if(bufferSize != outputSize || buffer == nullptr)
        {
            errorString = "Invaild parameters";
            return false;
        }

        buffers.insert(buffers.begin() + outputIndex, buffer);
        return true;
    }

    std::string errorString;

    private:

    struct NvInferDeleter 
    {
        template <typename T>
        void operator()(T* obj) const 
        {
            if (obj) 
            {
                obj->destroy();
            }
        }
    };

    class Logger : public nvinfer1::ILogger           
    {
        void log(Severity severity, const char* msg) noexcept override
        {
            // suppress info-level messages
            if (severity == Severity::kINFO)
                ROS_INFO("%s",msg);
            else if (severity == Severity::kVERBOSE)
                ROS_DEBUG("%s",msg);
            else if (severity == Severity::kWARNING)
                ROS_WARN("%s",msg);
            else
                ROS_ERROR("%s",msg);
        }
    } logger;

    std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, NvInferDeleter> context;

    std::vector<void*> buffers;
    cudaStream_t stream;

    size_t getDataTypeSize(nvinfer1::DataType type) 
    {
        switch (type) 
        {
            case nvinfer1::DataType::kFLOAT: return 4;   // 4 bytes for float
            case nvinfer1::DataType::kHALF:  return 2;   // 2 bytes for half-precision float
            case nvinfer1::DataType::kINT8:  return 1;   // 1 byte for int8
            case nvinfer1::DataType::kINT32: return 4;   // 4 bytes for int32
            case nvinfer1::DataType::kBOOL:  return 1;   // 1 byte for bool
            default: return 0;
        }
    }   

    size_t getVolume(const nvinfer1::Dims& dims) 
    {
        size_t volume = 1; // Start with a volume of 1
        for (int i = 0; i < dims.nbDims; ++i) 
        {
            if (dims.d[i] == -1) 
            {
                return 0;
            }
            volume *= dims.d[i];
        }
        return volume;
    }

    std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> loadOnnx(const std::string& fileName)
    {
        std::unique_ptr<nvinfer1::IBuilder, NvInferDeleter> builder{nvinfer1::createInferBuilder(logger)};
        if (builder == nullptr)
        {
            errorString = "Failed to create the network builder";
            return nullptr;
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        std::unique_ptr<nvinfer1::INetworkDefinition, NvInferDeleter> network{builder->createNetworkV2(explicitBatch)};
        if (network == nullptr)
        {
            errorString = "Failed to build the network";
            return nullptr;
        }
        
        std::unique_ptr<nvonnxparser::IParser, NvInferDeleter> parser{nvonnxparser::createParser(*network, logger)};
        if (parser == nullptr)
        {
            errorString = "Failed to create the network parser";
            return nullptr;
        }

        if(!parser->parseFromFile(fileName.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kVERBOSE)))
        {
            errorString = "Failed to parse the model file";
            return nullptr;
        }


        std::unique_ptr<nvinfer1::IBuilderConfig, NvInferDeleter> config{builder->createBuilderConfig()};
        if (config == nullptr)
        {
            errorString = "Failed to create builder config";
            return nullptr;
        }

        size_t totalMemory;
        cudaError_t error = cudaMemGetInfo(nullptr, &totalMemory);
        if (error != cudaSuccess) 
        {
            errorString = "Failed to get cuda memory info: " + std::string(cudaGetErrorString(error));
            return nullptr;
        }

        config->setMaxWorkspaceSize(totalMemory/4);

        std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> engine{builder->buildEngineWithConfig(*network, *config)};

        return std::move(engine);
    }

    std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> loadEngine(const std::string& fileName)
    {
        std::ifstream file(fileName, std::ios::binary);
        if (!file) 
        {
            errorString = "Model file " + fileName + " does not exist";
            return nullptr;
        }

        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engineData(fileSize);
        file.read(engineData.data(), fileSize);
        if (!file) 
        {
            errorString = "Failed to read the model file " + fileName;
            return nullptr;
        } 

        file.close();

        std::unique_ptr<nvinfer1::IRuntime, NvInferDeleter> runtime{nvinfer1::createInferRuntime(logger)};
        if (runtime == nullptr)
        {
            errorString = "Failed to create the runtime";
            return nullptr;
        }

        std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> engine{runtime->deserializeCudaEngine(engineData.data(), engineData.size())};

        return std::move(engine);
    }

    bool saveEngine(const std::string& fileName, const std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter>& engine)
    {
        //Serialize and save the model
        std::unique_ptr<nvinfer1::IHostMemory, NvInferDeleter> serializedModel{engine->serialize()};

        std::ofstream file(fileName, std::ios::binary);
        if (!file) 
        {
            errorString = "Failed to create the engine file " + fileName;
            return false;
        }

        // Write the serialized model to the file
        file.write(static_cast<const char*>(serializedModel->data()), serializedModel->size());
        if (!file) 
        {
            errorString = "Failed to write to the engine file" + fileName;
            return false;
        }   

        file.close();
        return true;
    }
};

class NNController: controller::Controller
{
    public:
    NNController() = delete;
    NNController(ros::NodeHandle& nodeHandle):controller::Controller(nodeHandle),
        imageBuffer(nullptr), steerAngle(nullptr)
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
            ROS_WARN("Invaild imageWidth. Using default value 1280");
        }

        if(imageHeight <= 0)
        {
            ROS_WARN("Invaild imageHeight. Using defaule value 720");
        }

        ROS_INFO("Set up tensorrt environment for model %s, this takes a while, please be patient", modelFile.c_str());
        infer = std::make_unique<Infer>(modelFile);

        ROS_INFO("-----------------------------------------------");
        ROS_INFO(" NN Configuration");
        ROS_INFO("-----------------------------------------------");
        ROS_INFO("%-20s | %-10s ", "Parameter", "Value");
        ROS_INFO("---------------------+------------+------------");
        ROS_INFO("%-20s | %-10d ", "Image width", imageWidth);
        ROS_INFO("%-20s | %-10d ", "Image height", imageHeight);
        ROS_INFO("%-20s | %-10d ", "Gray scale", grayScale);
        ROS_INFO("-----------------------------------------------");

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

        cudaError_t error = cudaMallocManaged(&imageBuffer, imageStep * imageHeight);
        if(error != cudaSuccess)
        {
            throw std::runtime_error("Failed to allocate memory for the input tensor: " + std::string(cudaGetErrorString(error)));
        }

        error = cudaMallocManaged(&steerAngle, sizeof(float));
        if(error != cudaSuccess)
        {
            throw std::runtime_error("Failed to allocate memory for the output tensor: " + std::string(cudaGetErrorString(error)));
        }

        imageMat =  cv::Mat(imageHeight, imageWidth, imageType, imageBuffer, imageStep);

        if(!infer->setInputBuffer(inputName, imageBuffer, imageStep * imageHeight))
        {
            throw std::runtime_error("Failed to assign the input tensor: " + infer->errorString);
        }

        if(!infer->setOutputBuffer(outputName, steerAngle, sizeof(float)))
        {
            throw std::runtime_error("Failed to assign the output tensor: " + infer->errorString);
        }
    }

    ~NNController()
    {
        if(imageBuffer != nullptr)
            cudaFree(imageBuffer);

        if(steerAngle != nullptr)
            cudaFree(steerAngle);
    }

    private:

    

    std::unique_ptr<Infer> infer;
    ros::Subscriber imageSub;
    cv::Mat imageMat;
    void* steerAngle;
    void* imageBuffer;
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






