// #include <ros/ros.h>
// #include <torch/script.h>
// #include <stdexcept>
// #include <cstdlib>
// #include <algorithm>
// #include <vector>
// #include <string>

// #ifdef __cplusplus
// extern "C" {
// #endif

// extern const char* PT_Version(void);

// #ifdef __cplusplus
// }
// #endif

// struct PTInferencer
// {
//     PTInferencer()
//     {
//         PT_Version();
//     }

//     ~PTInferencer()
//     {
//     }

//     torch::jit::script::Module model;
//     std::vector<torch::Tensor> inputTensors;
//     torch::Tensor outputTensor;
//     std::string errorString;
//     void** outputBuffer;
//     torch::Device device = torch::kCPU; // Default to CPU
// };

// extern "C" void* createInferencer(void* options)
// {
//     return new PTInferencer();
// }

// extern "C" void deleteInferencer(void* inferencerHandle)
// {
//     if (inferencerHandle != nullptr)
//     {
//         delete reinterpret_cast<PTInferencer*>(inferencerHandle);
//     }
// }

// extern "C" bool loadModel(void* inferencerHandle, const char* modelName)
// {
//     if (inferencerHandle == nullptr)
//         return false;

//     PTInferencer* inferencer = reinterpret_cast<PTInferencer*>(inferencerHandle);

//     try {
//         inferencer->model = torch::jit::load(modelName);
//         inferencer->model.to(inferencer->device); // Move model to CPU/GPU
//     } catch (const c10::Error& e) {
//         inferencer->errorString = "Failed to load model: " + std::string(e.what());
//         return false;
//     }

//     return true;
// }

// extern "C" unsigned getInputBuffer(void* inferencerHandle, void** buffer, int batch_size, int channels, int height, int width)
// {
//     if (inferencerHandle == nullptr)
//         return 0;

//     PTInferencer* inferencer = reinterpret_cast<PTInferencer*>(inferencerHandle);
//     torch::Tensor inputTensor = torch::zeros({batch_size, channels, height, width}, torch::kFloat32).to(inferencer->device);

//     inferencer->inputTensors.push_back(inputTensor);
//     *buffer = inputTensor.contiguous().data_ptr<float>();

//     return inputTensor.numel() * sizeof(float);
// }

// extern "C" unsigned getOutputBuffer(void* inferencerHandle, void** buffer)
// {
//     if (inferencerHandle == nullptr)
//         return 0;

//     PTInferencer* inferencer = reinterpret_cast<PTInferencer*>(inferencerHandle);

//     inferencer->outputBuffer = buffer;
//     return sizeof(float) * inferencer->outputTensor.numel();
// }

// extern "C" bool infer(void* inferencerHandle)
// {
//     if (inferencerHandle == nullptr)
//         return false;

//     PTInferencer* inferencer = reinterpret_cast<PTInferencer*>(inferencerHandle);

//     try {
//         std::vector<torch::jit::IValue> inputs;
//         inputs.push_back(inferencer->inputTensors[0]);

//         inferencer->outputTensor = inferencer->model.forward(inputs).toTensor();
//         inferencer->outputTensor = inferencer->outputTensor.contiguous();
//         *inferencer->outputBuffer = inferencer->outputTensor.data_ptr<float>();
//     } catch (const c10::Error& e) {
//         inferencer->errorString = "Inference failed: " + std::string(e.what());
//         return false;
//     }

//     return true;
// }

// extern "C" const char* getErrorString(void* inferencerHandle)
// {
//     if (inferencerHandle == nullptr)
//         return nullptr;

//     return reinterpret_cast<PTInferencer*>(inferencerHandle)->errorString.c_str();
// }

#include <ros/ros.h>
#include <torch/script.h>
#include <torch/cuda.h>
#include <c10/core/Device.h>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/json_parser.hpp>

#ifdef __cplusplus
extern "C" {
#endif

extern const char* PT_Version(void);

#ifdef __cplusplus
}
#endif

class PTInferencer
{
public:
    PTInferencer() : device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU){}

    ~PTInferencer(){}

    torch::jit::script::Module model;
    std::vector<torch::Tensor> inputTensors;
    torch::Tensor outputTensor;
    std::string errorString;
    void** outputBuffer;
    std::string modelName;
    torch::Device device;
};

extern "C" void* createInferencer(void* options)
{
    return new PTInferencer();
}

extern "C" void deleteInferencer(void* inferencerHandle)
{
    if (inferencerHandle != nullptr)
    {
        delete reinterpret_cast<PTInferencer*>(inferencerHandle);
    }
}

extern "C" bool loadModel(void* inferencerHandle, const char* modelName)
{
    if (inferencerHandle == nullptr)
        return false;

    PTInferencer* inferencer = reinterpret_cast<PTInferencer*>(inferencerHandle);
    inferencer -> modelName = modelName;
    try {
        inferencer->model = torch::jit::load(modelName);
        inferencer->model.to(inferencer->device); // Move model to CPU/GPU
    } catch (const c10::Error& e) {
        inferencer->errorString = "Failed to load model: " + std::string(e.what());
        return false;
    }

    return true;
}

extern "C" unsigned getInputBuffer(void* inferencerHandle, const char* inputName, void** buffer)
{

    if (inferencerHandle == nullptr)
        return 0;

    PTInferencer* inferencer = reinterpret_cast<PTInferencer*>(inferencerHandle);

    boost::filesystem::path modelPath(inferencer->modelName);
    boost::filesystem::path jsonPath = modelPath.replace_extension("json");

    boost::property_tree::ptree pTree;
    torch::Dtype modelDtype;
    int ModelInVals[3];
    try {
        boost::property_tree::read_json(jsonPath.string(), pTree);
        
        const boost::property_tree::ptree& input = pTree.get_child(inputName);


        std::string TorDType = input.get<std::string>("datatype");
      

        static const std::unordered_map<std::string, torch::Dtype> dtypeMap = {
                {"float", torch::kFloat32},
                {"double", torch::kFloat64},
                {"int", torch::kInt32},
                {"long", torch::kInt64}
        };

        auto it = dtypeMap.find(TorDType);
        modelDtype = it->second;


        auto inputVals = input.get_child("dimension");
        
        int index = 0;

        for(const auto& item : inputVals){
            ModelInVals[index] = item.second.get_value<int>();
            index++;
        }
    } catch (const boost::property_tree::json_parser::json_parser_error& e) {
        std::cerr << "Error reading JSON file: " << e.what() << std::endl;
        return 1;
    }
    // The hard coded argument is the batch size this is assumed to stay constant
    torch::Tensor inputTensor = torch::zeros({ModelInVals[0], ModelInVals[1], ModelInVals[2]}, modelDtype).to(inferencer->device);
    inferencer->inputTensors.push_back(inputTensor);
    *buffer = inputTensor.contiguous().data_ptr<float>();
    return inputTensor.numel() * sizeof(float);
}

extern "C" unsigned getOutputBuffer(void* inferencerHandle, const char* outputName, void** buffer)
{
    if (inferencerHandle == nullptr)
        return 0;

    PTInferencer* inferencer = reinterpret_cast<PTInferencer*>(inferencerHandle);

    boost::filesystem::path modelPath(inferencer->modelName);
    boost::filesystem::path jsonPath = modelPath.replace_extension("json");

    boost::property_tree::ptree pTree;
    torch::Dtype modelDtype;
    int ModelOutVals[1];
    try {
        boost::property_tree::read_json(jsonPath.string(), pTree);
        
        const boost::property_tree::ptree& output = pTree.get_child(outputName);


        std::string TorDType = output.get<std::string>("datatype");
      

        static const std::unordered_map<std::string, torch::Dtype> dtypeMap = {
                {"float", torch::kFloat32},
                {"double", torch::kFloat64},
                {"int", torch::kInt32},
                {"long", torch::kInt64}
        };

        modelDtype = dtypeMap.find(TorDType)->second;


        auto outputVals = output.get_child("dimension");
        
        int index = 0;

        for(const auto& item : outputVals){
            ModelOutVals[index] = item.second.get_value<int>();
            index++;
        }
    } catch (const boost::property_tree::json_parser::json_parser_error& e) {
        std::cerr << "Error reading JSON file: " << e.what() << std::endl;
        return 1;
    }
    // The hard coded argument is the batch size this is assumed to stay constant
    torch::Tensor outputTensor = torch::zeros({ModelOutVals[0]}, modelDtype).to(inferencer->device);

    inferencer->outputTensor = outputTensor;
    *buffer = inferencer->outputTensor.contiguous().data_ptr<float>();
    inferencer->outputBuffer = buffer;
    return outputTensor.numel() * sizeof(float);
    //inferencer->outputBuffer = buffer;
}

extern "C" bool infer(void* inferencerHandle)
{
    if (inferencerHandle == nullptr)
        return false;

    PTInferencer* inferencer = reinterpret_cast<PTInferencer*>(inferencerHandle);

    try {
        std::vector<torch::jit::IValue> inputs;

        inputs.push_back(inferencer->inputTensors[0]);

        inferencer->outputTensor = inferencer->model.forward(inputs).toTensor().contiguous();
        *inferencer->outputBuffer = inferencer->outputTensor.data_ptr<float>();
    } catch (const c10::Error& e) {
        inferencer->errorString = "Inference failed: " + std::string(e.what());
        return false;
    }

    return true;
}

extern "C" const char* getErrorString(void* inferencerHandle)
{
    if (inferencerHandle == nullptr)
        return nullptr;

    return reinterpret_cast<PTInferencer*>(inferencerHandle)->errorString.c_str();
}