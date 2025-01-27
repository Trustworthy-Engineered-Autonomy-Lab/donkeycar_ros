#include <ros/ros.h>
#include <cuda_runtime_api.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <boost/filesystem.hpp>

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

struct RTInferencer
{
    RTInferencer(): buffers(1, nullptr)
    {
        cudaError_t error = cudaStreamCreate(&stream);

        if(error != cudaSuccess)
        {
            throw std::runtime_error("Failed to create cuda stream: " + std::string(cudaGetErrorString(error)));
        }

    }

    ~RTInferencer()
    {
        cudaStreamDestroy(stream);

        for(const auto& buffer:buffers)
        {
            if(buffer != nullptr)
                cudaFree(buffer);
        }
    }

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

};


std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> loadOnnx(RTInferencer* inferencer, const std:: string& fileName)
{
    std::unique_ptr<nvinfer1::IBuilder, NvInferDeleter> builder{nvinfer1::createInferBuilder(inferencer->logger)};
    if (builder == nullptr)
    {
        throw std::runtime_error("Failed to create tensorrt network builder");
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition, NvInferDeleter> network{builder->createNetworkV2(explicitBatch)};
    if (network == nullptr)
    {
        throw std::runtime_error("Failed to build tensorrt network");
    }
    
    std::unique_ptr<nvonnxparser::IParser, NvInferDeleter> parser{nvonnxparser::createParser(*network, inferencer->logger)};
    if (parser == nullptr)
    {
        throw std::runtime_error("Failed to create onnx model parser");
    }

    if(!parser->parseFromFile(fileName.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kVERBOSE)))
    {
        inferencer->errorString = "Failed to parse onnx model file " + fileName;
        return nullptr;
    }


    std::unique_ptr<nvinfer1::IBuilderConfig, NvInferDeleter> config{builder->createBuilderConfig()};
    if (config == nullptr)
    {
        throw std::runtime_error("Failed to create tensorrt builder configuration");
    }

    size_t totalMemory;
    cudaError_t error = cudaMemGetInfo(nullptr, &totalMemory);
    if (error != cudaSuccess) 
    {
        throw std::runtime_error("Failed to get cuda memory info: " + std::string(cudaGetErrorString(error)));
    }

    config->setMaxWorkspaceSize(totalMemory/4);

    std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> engine{builder->buildEngineWithConfig(*network, *config)};
    if(engine == nullptr)
    {
        inferencer->errorString = "Failed to build tensorrt engine from onnx model " + fileName;
        return nullptr;
    }

    return std::move(engine);
}

std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> loadEngine(RTInferencer* inferencer,const std::string& fileName)
{
    std::ifstream file(fileName, std::ios::binary);
    if (!file) 
    {
        inferencer->errorString = "Model file " + fileName + " does not exist";
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);
    if (!file) 
    {
        inferencer->errorString = "Failed to read the model file " + fileName;
        return nullptr;
    } 

    file.close();

    std::unique_ptr<nvinfer1::IRuntime, NvInferDeleter> runtime{nvinfer1::createInferRuntime(inferencer->logger)};
    if (runtime == nullptr)
    {
        throw std::runtime_error("Failed to create tensorrt runtime");
    }

    std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> engine{runtime->deserializeCudaEngine(engineData.data(), engineData.size())};
    if (engine == nullptr)
    {
        inferencer->errorString = "Failed to deserialize tensorrt engine file " + fileName;
        return nullptr;
    }

    return std::move(engine);
}

bool saveEngine(RTInferencer* inferencer, const std::string& fileName)
{
    //Serialize and save the model
    std::unique_ptr<nvinfer1::IHostMemory, NvInferDeleter> serializedModel{inferencer->engine->serialize()};

    std::ofstream file(fileName, std::ios::binary);
    if (!file) 
    {
        inferencer->errorString = "Failed to create the engine file " + fileName;
        return false;
    }

    // Write the serialized model to the file
    file.write(static_cast<const char*>(serializedModel->data()), serializedModel->size());
    if (!file) 
    {
        inferencer->errorString = "Failed to write to the engine file" + fileName;
        return false;
    }   

    file.close();
    return true;
}

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

void* allocBuffer(RTInferencer* inferencer, int index, size_t size)
{
    if(index + 1 <= inferencer->buffers.size())
    {
        if(inferencer->buffers[index] != nullptr)
        {
            return inferencer->buffers[index];
        }
    }
    else
    {
        inferencer->buffers.resize(index + 1, nullptr);
    }

    void* buffer;
    cudaError_t error = cudaMallocManaged(&buffer, size);
    if(error != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate memory for the input tensor: " + std::string(cudaGetErrorString(error)));
    }

    inferencer->buffers[index] = buffer;
    return buffer;
}

extern "C" void* createInferencer(void* options)
{
    RTInferencer* inferencer = new RTInferencer();

    return inferencer;
}

extern "C" void deleteInferencer(void* inferencerHandle)
{
    if(inferencerHandle != nullptr)
    {
        delete reinterpret_cast<RTInferencer*>(inferencerHandle);
    }
}

extern "C" bool loadModel(void* inferencerHandle, const char* modelName)
{
    if(inferencerHandle == nullptr)
        return false;
    
    RTInferencer* inferencer = reinterpret_cast<RTInferencer*>(inferencerHandle);

    bool needSerialize = false;
    boost::filesystem::path filePath(modelName);

    std::string fileExt = filePath.extension().string();

    if(fileExt == ".onnx")
    {
        boost::filesystem::path engineFilePath = filePath;
        engineFilePath.replace_extension("engine");

        if(boost::filesystem::exists(engineFilePath))
        {
            inferencer->engine = loadEngine(inferencer,engineFilePath.c_str());
        }


        if(inferencer->engine == nullptr)
        {
            inferencer->engine = loadOnnx(inferencer,filePath.string());
            if(inferencer->engine != nullptr)
                saveEngine(inferencer ,engineFilePath.string());
        }
    }
    else if(fileExt == ".engine")
    {
        inferencer->engine = loadEngine(inferencer, filePath.string());
    }
    else
    {
        inferencer->errorString = "Unsupported model file format " + fileExt;
        return false;
    }


    if (inferencer->engine == nullptr)
    {
        return false;
    }

    inferencer->context.reset(inferencer->engine->createExecutionContext());

    if(inferencer->context == nullptr)
    {
        throw std::runtime_error("Failed to create tensorrt execution context");
    }
    return true;
}

extern "C" unsigned getInputBuffer(void* inferencerHandle, const char* inputName, void** buffer)
{
    if(inferencerHandle == nullptr)
        return false;
    
    RTInferencer* inferencer = reinterpret_cast<RTInferencer*>(inferencerHandle);

    int inputIndex = inferencer->engine->getBindingIndex(inputName);

    if(inputIndex == -1)
    {
        inferencer->errorString = "Invaild input tensor name " + std::string(inputName);
        return 0;
    }

    nvinfer1::Dims inputDims = inferencer->engine->getBindingDimensions(inputIndex);
    nvinfer1::DataType inputDataType = inferencer->engine->getBindingDataType(inputIndex);

    size_t inputSize = getVolume(inputDims) * getDataTypeSize(inputDataType);

    *buffer = allocBuffer(inferencer,inputIndex, inputSize);

    if(buffer == nullptr)
        return 0;

    return inputSize;
} 

extern "C" unsigned getOutputBuffer(void* inferencerHandle, const char* outputName, void** buffer)
{
    if(inferencerHandle == nullptr)
        return false;
    
    RTInferencer* inferencer = reinterpret_cast<RTInferencer*>(inferencerHandle);

    int outputIndex = inferencer->engine->getBindingIndex(outputName);
    if(outputIndex == -1)
    {
        inferencer->errorString = "Invaild output tensor name " + std::string(outputName);
        return false;
    }

    nvinfer1::Dims outputDims = inferencer->engine->getBindingDimensions(outputIndex);
    nvinfer1::DataType outputDataType = inferencer->engine->getBindingDataType(outputIndex); 

    size_t outputSize = getVolume(outputDims) * getDataTypeSize(outputDataType);

    *buffer = allocBuffer(inferencer,outputIndex, outputSize);

    if(buffer == nullptr)
        return 0;

    return outputSize;
} 

extern "C" bool infer(void* inferencerHandle)
{
    if(inferencerHandle == nullptr)
        return false;
    
    RTInferencer* inferencer = reinterpret_cast<RTInferencer*>(inferencerHandle);

    if(!inferencer->context->enqueueV2(inferencer->buffers.data(), inferencer->stream, nullptr))
    {
        inferencer->errorString = "Failed to enqueue the stream";
        return false;
    }

    cudaStreamSynchronize(inferencer->stream);
    return true;
}

extern "C" const char* getErrorString(void* inferencerHandle)
{
    if(inferencerHandle == nullptr)
        return nullptr;

    return reinterpret_cast<RTInferencer*>(inferencerHandle)->errorString.c_str();
}