#include <string>

#include <ros/ros.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <boost/filesystem.hpp>

namespace inferencer
{
    class Inferencer
    {
        public:
        Inferencer(){};
        virtual ~Inferencer(){}

        virtual size_t getInputBuffer(const std::string& inputName, void** bufferPtr){return 0;}
        virtual size_t getOutputBuffer(const std::string& inputName, void** bufferPtr){return 0;}

        virtual bool infer(){return false;}
        std::string getErrorString(){return errorString;}

        protected:
        std::string errorString;
    };

    class RTInferencer:public Inferencer
    {
        public:
        RTInferencer() = delete;
        RTInferencer(const boost::filesystem::path& filePath);
        ~RTInferencer();

        bool infer();
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
    
}