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
