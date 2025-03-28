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
private:

    torch::jit::script::Module model;
    std::vector<torch::Tensor> inputTensors;
    torch::Tensor outputTensor;
    std::string errorString;
    void** outputBuffer;
    std::string modelName;
    torch::Device device;

    std::vector<int64_t> ModelInVals;
    std::vector<int64_t> ModelOutVals;
    torch::Dtype inputDtype;
    torch::Dtype outputDtype;

public:
    PTInferencer()
            : device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
              inputBufferPtr(nullptr), outputBufferPtr(nullptr){}

    ~PTInferencer(){}

    void* createInferencer(void* options){
        return new PTInferencer();
    }

    void deleteInferencer(){
       delete this;
    }
    bool loadModel(const char* modelName)
    {
        this-> modelName = modelName;
        try {
            this->model = torch::jit::load(modelName);
            this->model.to(this->device); // Move model to CPU/GPU
            if(!parseModelJson()){
                return false;
            }
        } catch (const c10::Error& e) {
            this->errorString = "Failed to load model: " + std::string(e.what());
            return false;
        }

        return true;
    }
    unsigned getInputBuffer(void** buffer)
    {
        /*boost::filesystem::path modelPath(this->modelName);
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
         */
        torch::Tensor inputTensor = torch::zeros({this->ModelInVals[0], this->ModelInVals[1], this->ModelInVals[2]}, modelDtype).to(this->device);
        this->inputTensors.push_back(inputTensor);
        *buffer = inputTensor.contiguous().data_ptr<float>();
        return inputTensor.numel() * sizeof(float);
    }
    unsigned getOutputBuffer(void** buffer)
    {
        /*
        boost::filesystem::path modelPath(this->modelName);
        boost::filesystem::path jsonPath = modelPath.replace_extension("json");

        boost::property_tree::ptree pTree;
        torch::Dtype modelDtype;
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
         */
        this->outputTensor = torch::zeros({ModelOutVals[0]}, modelDtype).to(this->device);
        *buffer = this->outputTensor.contiguous().data_ptr<float>();
        *this->outputBuffer = buffer;
        return outputTensor.numel() * sizeof(float);
        //inferencer->outputBuffer = buffer;
    }

    bool infer()
    {
        try {
            std::vector<torch::jit::IValue> inputs;

            inputs.push_back(this->inputTensors[0]);

            this->outputTensor = this->model.forward(inputs).toTensor().contiguous();
            *this->outputBuffer = this->outputTensor.data_ptr<float>();
        }
        catch (const c10::Error& e) {
            this->errorString = "Inference failed: " + std::string(e.what());
            return false;
        }

        return true;
    }

    const char* getErrorString(void* )
    {
        return this->errorString.c_str();
    }

    bool parseModelJson(){
        boost::filesystem::path jsonPath(this->modelName);
        jsonPath.replace_extension("json");

        boost::property_tree::ptree pTree;
        try {
            boost::property_tree::read_json(jsonPath.string(), pTree);

            auto input = pTree.get_child("input");
            auto output = pTree.get_child("output");

            std::string inType = input.get<std::string>("datatype");
            std::string outType = output.get<std::string>("datatype");

            static const std::unordered_map<std::string, torch::Dtype> dtypeMap = {
                    {"float", torch::kFloat32},
                    {"double", torch::kFloat64},
                    {"int", torch::kInt32},
                    {"long", torch::kInt64}
            };

            this->inputDtype = dtypeMap.at(inType);
            this->outputDtype = dtypeMap.at(outType);

            for (auto& dim : input.get_child("dimension"))
                this->ModelInVals.push_back(dim.second.get_value<int>());
            for (auto& dim : output.get_child("dimension"))
                this->ModelOutVals.push_back(dim.second.get_value<int>());

            return true;
        } catch (...) {
            return false;
        }
    }
};