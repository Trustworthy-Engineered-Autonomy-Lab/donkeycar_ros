#include <inferencer/inferencer.h>

#include <stdexcept>
#include <cstdlib>
#include <algorithm>

#include <tensorflow/core/platform/status.h>
#include <tensorflow/core/platform/errors.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/logging.h>

#include <boost/process/environment.hpp>

#ifdef __cplusplus
extern "C" {
#endif

extern const char* TF_Version(void);

#ifdef __cplusplus
}
#endif

namespace inferencer
{
    TFInferencer::TFInferencer(const ros::NodeHandle& nodeHandle):Inferencer(nodeHandle)
    {
        // boost::process::environment env = boost::this_process::environment();
        // env["TF_CPP_MIN_LOG_LEVEL"] = "1";
        
        TF_Version();

    }

    TFInferencer::~TFInferencer()
    {

    }

    bool TFInferencer::loadModel(const std::string& modelName)
    {
        tensorflow::SessionOptions sessionOptions;
        tensorflow::RunOptions runOptions;
        
        tensorflow::Status status = tensorflow::LoadSavedModel(sessionOptions, runOptions, modelName, {"serve"}, &modelBundle);

        if (!status.ok()) 
        {
            errorString = status.ToString();
            return false;
        }

        const tensorflow::MetaGraphDef& graphDef = modelBundle.meta_graph_def;

        const auto& signatureDefMap = graphDef.signature_def();
        
        auto it = signatureDefMap.find("serving_default");
        if(it == signatureDefMap.end())
        {
            errorString = "Failed to find the default signature";
            return false;
        }

        signature = it->second;
        return true;
    }

    size_t TFInferencer::getInputBuffer(const std::string& inputName, void** bufferPtr)
    {
        auto it = signature.inputs().find(inputName);
        if(it == signature.inputs().end())
        {
            std::ostringstream errorStream;
            errorStream << "Invalid input tensor name: " << inputName << ". Valid names are: ";

            // Iterate over all valid input tensor names
            for (const auto& input : signature.inputs())
            {
                errorStream << input.first << " ";
            }

            errorString = errorStream.str();
            return 0;
        }

        const tensorflow::TensorInfo& tensorInfo = it->second;
        const std::string& tensorName = tensorInfo.name();

        auto it1 = std::find_if(
            inputTensorPairs.begin(),
            inputTensorPairs.end(),
            [&tensorName](const std::pair<std::string, tensorflow::Tensor>& pair) {
                return pair.first == tensorName; // Compare the tensor name
            });
        
        if(it1 != inputTensorPairs.end())
        {
            int index = std::distance(inputTensorPairs.begin(), it1);
            *bufferPtr = inputTensorPairs[index].second.data();
            return inputTensorPairs[index].second.tensor_data().size();
        }

        tensorflow::TensorShapeProto shapeProto = tensorInfo.tensor_shape();
        if(shapeProto.dim(0).size() == -1)
        {
            ROS_INFO("Dynamic batch size detected. Will set to one");
            shapeProto.mutable_dim(0)->set_size(1);
        }

        ROS_INFO_STREAM("Input tensor shape " + shapeProto.DebugString());

        tensorflow::TensorShape shape(shapeProto);

        tensorflow::Tensor inputTensor = tensorflow::Tensor(tensorInfo.dtype(), shape);

        if (!inputTensor.IsInitialized())
        {
            errorString = "Tensor is not initialized";
            return 0;
        }

        size_t inputByteSize = inputTensor.tensor_data().size();

        // *bufferPtr = const_cast<void*>(reinterpret_cast<const void*>(inputTensor.tensor_data().data()));

        inputTensorPairs.emplace_back(tensorName, std::move(inputTensor));

        *bufferPtr = inputTensorPairs.back().second.data();

        return inputByteSize;
    }

    size_t TFInferencer::getOutputBuffer(const std::string& outputName, void** bufferPtr)
    {   
        auto it = signature.outputs().find(outputName);
        if(it == signature.outputs().end())
        {
            std::ostringstream errorStream;
            errorStream << "Invalid output tensor name: " << outputName << ". Valid names are: ";

            // Iterate over all valid input tensor names
            for (const auto& output : signature.outputs())
            {
                errorStream << output.first << " ";
            }
            errorString = errorStream.str();
            return 0;
        }

        const tensorflow::TensorInfo& tensorInfo = it->second;
        const std::string& tensorName = tensorInfo.name();

        auto it1 = std::find(outputTensorNames.begin(),outputTensorNames.end(), tensorName);
        if(it1 != outputTensorNames.end())
        {
            int index = std::distance(outputTensorNames.begin(), it1);
            return outputTensors[index].tensor_data().size();
        }

        tensorflow::TensorShapeProto shapeProto = tensorInfo.tensor_shape();
        if(shapeProto.dim(0).size() == -1)
        {
            ROS_INFO("Dynamic batch size detected. Will set to one");
            shapeProto.mutable_dim(0)->set_size(1);
        }

        ROS_INFO_STREAM("Output tensor shape " + shapeProto.DebugString());
        tensorflow::TensorShape shape(shapeProto);

        tensorflow::Tensor outputTensor = tensorflow::Tensor(tensorInfo.dtype(), shape);

        if (!outputTensor.IsInitialized())
        {
            errorString = "Tensor is not initialized";
            return 0;
        }

        size_t outputByteSize = outputTensor.tensor_data().size();

        outputBuffer = bufferPtr;

        outputTensorNames.emplace_back(tensorName);
        outputTensors.emplace_back(std::move(outputTensor));

        return outputByteSize;
    }

    bool TFInferencer::infer()
    {
        tensorflow::Status status = modelBundle.GetSession()->Run(inputTensorPairs, outputTensorNames, {}, &outputTensors);
        if (!status.ok())
        {
            errorString = "Failed to run inference " + status.ToString();
            return false;
        }

        *outputBuffer = const_cast<void*>(reinterpret_cast<const void*>(outputTensors[0].tensor_data().data()));
        // ROS_INFO("Run inference successfully %X", data);

        return true;
    }

}