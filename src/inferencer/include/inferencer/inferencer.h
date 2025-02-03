#include <boost/function.hpp>

#include <dlfcn.h>

#include <string>

class Inferencer
{
    public:
    Inferencer(const std::string& backend):
        createInferencerFunc(nullptr), 
        deleteInferencerFunc(nullptr),
        inferencer(nullptr),
        dllHandle(nullptr)
    {
        std::string dllName = "lib" + backend + "_inferencer.so";
        dllHandle = dlopen(dllName.c_str(), RTLD_LAZY); 

        if(dllHandle == nullptr)
        {
            throw std::runtime_error("Failed to open shared library " + dllName + ": " + dlerror());
        }

        createInferencerFunc = reinterpret_cast<CreateInferencerFunc>(dlsym(dllHandle,"createInferencer"));
        deleteInferencerFunc = reinterpret_cast<DeleteInferencerFunc>(dlsym(dllHandle,"deleteInferencer"));

        LoadModelFunc loadModelFunc = reinterpret_cast<LoadModelFunc>(dlsym(dllHandle,"loadModel"));
        GetInputBufferFunc getInputBufferFunc = reinterpret_cast<GetInputBufferFunc>(dlsym(dllHandle,"getInputBuffer"));
        GetOutputBufferFunc getOutputBufferFunc = reinterpret_cast<GetOutputBufferFunc>(dlsym(dllHandle,"getOutputBuffer"));
        InferFunc inferFunc = reinterpret_cast<InferFunc>(dlsym(dllHandle,"infer"));
        GetErrorStringFunc getErrorStringFunc = reinterpret_cast<GetErrorStringFunc>(dlsym(dllHandle,"getErrorString"));

        if (!createInferencerFunc || !deleteInferencerFunc || !loadModelFunc || 
                !getInputBufferFunc || !getOutputBufferFunc || !inferFunc || !getErrorStringFunc)
        {
            throw std::runtime_error("Failed to load function symbols from " + dllName + ": " + dlerror());
        }

        inferencer = createInferencerFunc(nullptr);
        if(inferencer == nullptr)
        {
            throw std::runtime_error("Failed to create inferencer");
        }


        loadModel = boost::bind(loadModelFunc, inferencer, boost::placeholders::_1);
        getInputBuffer = boost::bind(getInputBufferFunc, inferencer, boost::placeholders::_1, boost::placeholders::_2);
        getOutputBuffer = boost::bind(getOutputBufferFunc, inferencer, boost::placeholders::_1, boost::placeholders::_2);
        infer = boost::bind(inferFunc, inferencer);
        getErrorString = boost::bind(getErrorStringFunc, inferencer);
    }

    ~Inferencer()
    {
        if(inferencer != nullptr)
        {
            deleteInferencerFunc(inferencer);
        }

        if(dllHandle != nullptr)
        {
            dlclose(dllHandle);
        }
    }

    boost::function<bool(const char*)> loadModel;
    boost::function<unsigned(const char*, void**)> getInputBuffer;
    boost::function<unsigned(const char*, void**)> getOutputBuffer;
    boost::function<bool()> infer;
    boost::function<const char*()> getErrorString;

    private:

    using CreateInferencerFunc = void*(*)(void*);
    using DeleteInferencerFunc = void (*) (void*);
    using LoadModelFunc = bool(*)(void*, const char*);
    using GetInputBufferFunc = unsigned (*)(void* , const char* , void** );
    using GetOutputBufferFunc = unsigned (*)(void* , const char*, void** );
    using InferFunc = bool(*)(void*);
    using GetErrorStringFunc = const char* (*) (void*);

    CreateInferencerFunc createInferencerFunc;
    DeleteInferencerFunc deleteInferencerFunc;

    void* dllHandle;
    void* inferencer;

};
