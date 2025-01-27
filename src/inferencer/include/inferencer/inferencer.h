#include <string>

#include <ros/ros.h>


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

        private:
        const ros::NodeHandle& nh;
    };
    
}