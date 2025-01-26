#include <string>

#include <ros/ros.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <boost/filesystem.hpp>

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/cc/saved_model/loader.h>


namespace inferencer
{
    class Inferencer
    {
        public:
        Inferencer() = delete;
        Inferencer(const ros::NodeHandle& nodeHandle):nh(nodeHandle){}
        virtual ~Inferencer(){}

        // virtual bool init();
        virtual bool loadModel(const std::string& modelName){return false;}
        virtual size_t getInputBuffer(const std::string& inputName, void** bufferPtr){return 0;}
        virtual size_t getOutputBuffer(const std::string& inputName, void** bufferPtr){return 0;}

        virtual bool infer(){return false;}
        const std::string& getErrorString(){return errorString;}

        protected:
        std::string errorString;
        // template <typename T>
        // T getParam(const std::string& paramName, const T& defaultValue)
        // {
        //     return nh.param<T>(paramName, paramValue, defaultValue);
        // }

        private:
        const ros::NodeHandle& nh;
    };

    class RTInferencer: public Inferencer
    {
        public:
        RTInferencer() = delete;
        RTInferencer(const ros::NodeHandle& nodeHandle);
        ~RTInferencer();

        bool infer();
        bool loadModel(const std::string& modelName);
        size_t getInputBuffer(const std::string& inputName, void** bufferPtr);
        size_t getOutputBuffer(const std::string& outputName, void** bufferPtr);

        

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
        std::string errorString;

        void* allocBuffer(int index, size_t size);
        size_t getDataTypeSize(nvinfer1::DataType type);
        size_t getVolume(const nvinfer1::Dims& dims) ;
        std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> loadOnnx(const std::string& fileName);
        std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> loadEngine(const std::string& fileName);
        bool saveEngine(const std::string& fileName, const std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter>& engine);
    };

    class TFInferencer: public Inferencer
    {
        public:
        TFInferencer() = delete;
        TFInferencer(const ros::NodeHandle& nodeHandle);
        ~TFInferencer();

        bool infer();
        bool loadModel(const std::string& modelName);
        size_t getInputBuffer(const std::string& inputName, void** bufferPtr);
        size_t getOutputBuffer(const std::string& outputName, void** bufferPtr);

        private:
        tensorflow::SavedModelBundle modelBundle;
        tensorflow::SignatureDef signature;
        std::vector<std::pair<std::string,tensorflow::Tensor>> inputTensorPairs;
        std::vector<std::string> outputTensorNames;
        std::vector<tensorflow::Tensor> outputTensors;

        void** outputBuffer;
    };
    
}