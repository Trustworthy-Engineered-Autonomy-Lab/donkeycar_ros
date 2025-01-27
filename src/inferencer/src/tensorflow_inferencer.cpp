#include <ros/ros.h>

#include <stdexcept>
#include <cstdlib>
#include <algorithm>

#include <tensorflow/core/platform/status.h>
#include <tensorflow/core/platform/errors.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/cc/saved_model/loader.h>

#include <boost/process/environment.hpp>
#include <boost/filesystem.hpp>

#ifdef __cplusplus
extern "C" {
#endif

extern const char* TF_Version(void);

#ifdef __cplusplus
}
#endif

struct TFInferencer
{
    TFInferencer()
    {
        // boost::process::environment env = boost::this_process::environment();
        // env["TF_CPP_MIN_LOG_LEVEL"] = "1";
        
        TF_Version();

    }

    ~TFInferencer()
    {

    }

    tensorflow::SavedModelBundle modelBundle;
    tensorflow::SignatureDef signature;
    std::vector<std::pair<std::string,tensorflow::Tensor>> inputTensorPairs;
    std::vector<std::string> outputTensorNames;
    std::vector<tensorflow::Tensor> outputTensors;

    std::string errorString;

    void** outputBuffer;
};

extern "C" void* createInferencer(void* options)
{
    return new TFInferencer();
}

extern "C" void deleteInferencer(void* inferencerHandle)
{
    if(inferencerHandle != nullptr)
    {
        delete reinterpret_cast<TFInferencer*>(inferencerHandle);
    }
}

extern "C" bool loadModel(void* inferencerHandle, const char* modelName)
{
    if(inferencerHandle == nullptr)
        return false;
    
    TFInferencer* inferencer = reinterpret_cast<TFInferencer*>(inferencerHandle);

    tensorflow::SessionOptions sessionOptions;
    tensorflow::RunOptions runOptions;
    
    tensorflow::Status status = tensorflow::LoadSavedModel(sessionOptions, runOptions, {modelName}, {"serve"}, &inferencer->modelBundle);

    if (!status.ok()) 
    {
        inferencer->errorString = status.ToString();
        return false;
    }

    const tensorflow::MetaGraphDef& graphDef = inferencer->modelBundle.meta_graph_def;

    const auto& signatureDefMap = graphDef.signature_def();
    
    auto it = signatureDefMap.find("serving_default");
    if(it == signatureDefMap.end())
    {
        inferencer->errorString = "Failed to find the default signature";
        return false;
    }

    inferencer->signature = it->second;
    return true;
}

extern "C" unsigned getInputBuffer(void* inferencerHandle, const char* inputName, void** buffer)
{
    if(inferencerHandle == nullptr)
        return false;
    
    TFInferencer* inferencer = reinterpret_cast<TFInferencer*>(inferencerHandle);

    auto it = inferencer->signature.inputs().find(inputName);
    if(it == inferencer->signature.inputs().end())
    {
        std::ostringstream errorStream;
        errorStream << "Invalid input tensor name: " << inputName << ". Valid names are: ";

        // Iterate over all valid input tensor names
        for (const auto& input : inferencer->signature.inputs())
        {
            errorStream << input.first << " ";
        }

        inferencer->errorString = errorStream.str();
        return 0;
    }

    const tensorflow::TensorInfo& tensorInfo = it->second;
    const std::string& tensorName = tensorInfo.name();

    auto it1 = std::find_if(
        inferencer->inputTensorPairs.begin(),
        inferencer->inputTensorPairs.end(),
        [&tensorName](const std::pair<std::string, tensorflow::Tensor>& pair) {
            return pair.first == tensorName; // Compare the tensor name
        });
    
    if(it1 != inferencer->inputTensorPairs.end())
    {
        int index = std::distance(inferencer->inputTensorPairs.begin(), it1);
        *buffer = inferencer->inputTensorPairs[index].second.data();
        return inferencer->inputTensorPairs[index].second.tensor_data().size();
    }

    tensorflow::TensorShapeProto shapeProto = tensorInfo.tensor_shape();
    if(shapeProto.dim(0).size() == -1)
    {
        // ROS_INFO("Dynamic batch size detected. Will set to one");
        shapeProto.mutable_dim(0)->set_size(1);
    }

    // ROS_INFO_STREAM("Input tensor shape " + shapeProto.DebugString());

    tensorflow::TensorShape shape(shapeProto);

    tensorflow::Tensor inputTensor = tensorflow::Tensor(tensorInfo.dtype(), shape);

    if (!inputTensor.IsInitialized())
    {
        inferencer->errorString = "Tensor is not initialized";
        return 0;
    }

    size_t inputByteSize = inputTensor.tensor_data().size();

    inferencer->inputTensorPairs.emplace_back(tensorName, std::move(inputTensor));

    *buffer = inferencer->inputTensorPairs.back().second.data();

    return inputByteSize;
} 

extern "C" unsigned getOutputBuffer(void* inferencerHandle, const char* outputName, void** buffer)
{
    if(inferencerHandle == nullptr)
        return false;
    
    TFInferencer* inferencer = reinterpret_cast<TFInferencer*>(inferencerHandle);

    auto it = inferencer->signature.outputs().find(outputName);
    if(it == inferencer->signature.outputs().end())
    {
        std::ostringstream errorStream;
        errorStream << "Invalid output tensor name: " << outputName << ". Valid names are: ";

        // Iterate over all valid input tensor names
        for (const auto& output : inferencer->signature.outputs())
        {
            errorStream << output.first << " ";
        }
        inferencer->errorString = errorStream.str();
        return 0;
    }

    const tensorflow::TensorInfo& tensorInfo = it->second;
    const std::string& tensorName = tensorInfo.name();

    auto it1 = std::find(inferencer->outputTensorNames.begin(), inferencer->outputTensorNames.end(), tensorName);
    if(it1 != inferencer->outputTensorNames.end())
    {
        int index = std::distance(inferencer->outputTensorNames.begin(), it1);
        return inferencer->outputTensors[index].tensor_data().size();
    }

    tensorflow::TensorShapeProto shapeProto = tensorInfo.tensor_shape();
    if(shapeProto.dim(0).size() == -1)
    {
        // ROS_INFO("Dynamic batch size detected. Will set to one");
        shapeProto.mutable_dim(0)->set_size(1);
    }

    // ROS_INFO_STREAM("Output tensor shape " + shapeProto.DebugString());
    tensorflow::TensorShape shape(shapeProto);

    tensorflow::Tensor outputTensor = tensorflow::Tensor(tensorInfo.dtype(), shape);

    if (!outputTensor.IsInitialized())
    {
        inferencer->errorString = "Tensor is not initialized";
        return 0;
    }

    size_t outputByteSize = outputTensor.tensor_data().size();

    inferencer->outputBuffer = buffer;

    inferencer->outputTensorNames.emplace_back(tensorName);
    inferencer->outputTensors.emplace_back(std::move(outputTensor));

    return outputByteSize;
} 

extern "C" bool infer(void* inferencerHandle)
{
    if(inferencerHandle == nullptr)
        return false;
    
    TFInferencer* inferencer = reinterpret_cast<TFInferencer*>(inferencerHandle);

    tensorflow::Status status = inferencer->modelBundle.GetSession()->Run(inferencer->inputTensorPairs, inferencer->outputTensorNames, {}, &inferencer->outputTensors);
    if (!status.ok())
    {
        inferencer->errorString = "Failed to run inference " + status.ToString();
        return false;
    }

    *inferencer->outputBuffer = const_cast<void*>(reinterpret_cast<const void*>(inferencer->outputTensors[0].tensor_data().data()));

    return true;
}

extern "C" const char* getErrorString(void* inferencerHandle)
{
    if(inferencerHandle == nullptr)
            return nullptr;
    
    return reinterpret_cast<TFInferencer*>(inferencerHandle)->errorString.c_str();
}
