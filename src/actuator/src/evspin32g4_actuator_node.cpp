#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <dynamic_reconfigure/server.h>

#include <actuator/actuator.h>
#include <actuator/EVSPIN32G4ActuatorConfig.h>

#include <string>
#include <cerrno>
#include <cstring>

#include <boost/thread.hpp>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>

class EVSPIN32G4
{
    public:
    EVSPIN32G4(std::string& deviceName, unsigned address = 0x08):
        fileHandle(-1)
    {
        Open(deviceName, address);
    }

    EVSPIN32G4()=default;

    ~EVSPIN32G4()
    {
        Close();
    }

    bool Open(std::string& deviceName, unsigned address = 0x08)
    {
        fileHandle = open(deviceName.c_str(), O_RDWR);

        if (fileHandle < 0) 
        {
            errorString = "Failed to open the I2C bus: " + deviceName;
            return false;
        }

        if (ioctl(fileHandle, I2C_SLAVE, address) < 0) 
        {
            errorString = "Failed to acquire bus access to " + deviceName;
            return false;
        }

        return true;
    }

    void Close()
    {
        if(fileHandle != -1)
        {
            close(fileHandle);
            fileHandle = -1;
        }
    }

    const std::string& getErrorString()
    {
        return errorString;
    }

    bool send_command(int16_t rpm, uint8_t angle)
    {
        uint8_t buffer[3] = {(uint8_t)(rpm >> 8), (uint8_t)rpm, angle};
        if (write(fileHandle, buffer, sizeof(buffer)) != sizeof(buffer))
        {
            errorString = strerror(errno);
            return false;
        }

        return true;
    }

    bool read_rpm(int16_t* rpm)
    {
        if (read(fileHandle, (uint8_t*)rpm, sizeof(int16_t)) != sizeof(int16_t)) 
        {
            errorString = strerror(errno);
            return false;
        }
        return true;
    }
    

    private:

    int fileHandle;
    std::string errorString;
};

class EVSPIN32G4Actuator: public actuator::Actuator
{
    public:
    EVSPIN32G4Actuator(ros::NodeHandle& nodeHandle):server(nodeHandle),
        nh(nodeHandle),
        status(Status::WAITING),
        evspin32g4{std::make_unique<EVSPIN32G4>()}
    {

        steerMinAngle = 30;
        steerMaxAngle = 120;
        steerMidAngle = 90;

        motorMinRPM = -5000;
        motorMaxRPM = 5000;

        server.setCallback(boost::bind(&EVSPIN32G4Actuator::serverCallback,this,boost::placeholders::_1,boost::placeholders::_2));
        timer = nodeHandle.createTimer(ros::Duration(1), boost::bind(&EVSPIN32G4Actuator::timerCallback, this, boost::placeholders::_1));
    }

    void actuate(float throttle, float steer)
    {
        if(status == Status::RUNNING)
        {
            uint8_t steerAngle;
            if(steer > 0)
            {
                steerAngle = steer * static_cast<float>(steerMaxAngle - steerMidAngle) + steerMidAngle;
            }
            else
            {
                steerAngle = steer * static_cast<float>(steerMidAngle - steerMinAngle) + steerMidAngle;
            }

            int16_t motorRPM = throttle / 2.0f * static_cast<float>(motorMaxRPM - motorMinRPM);

            ROS_DEBUG("Steer angle %d, Motor RPM %d", steerAngle, motorRPM);

            if(!evspin32g4->send_command(motorRPM, steerAngle))
            {
                ROS_ERROR("Failed to send command to the EVSPIN32G4 board: %s. Will retry", evspin32g4->getErrorString().c_str());
                status = Status::INITING;
                evspin32g4->Close();
                return;
            }
        }
    }

    private:

    enum class Status
    {
        WAITING,
        INITING,
        RUNNING,
    };

    std::unique_ptr<EVSPIN32G4> evspin32g4;
    std::string busDevice;
    ros::Timer timer;
    Status status;

    int16_t motorMinRPM;
    int16_t motorMaxRPM;
    int16_t motorRPM;

    uint8_t steerMinAngle;
    uint8_t steerMaxAngle;
    uint8_t steerMidAngle;

    ros::NodeHandle& nh;

    dynamic_reconfigure::Server<actuator::EVSPIN32G4ActuatorConfig> server;

    inline bool checkPWMChannel(int channel){return channel >= 0 && channel <= 15;}
    inline bool checkPWMPW(int minPW, int midPW, int maxPW){return minPW > 0 && midPW > minPW && maxPW > midPW;}
    inline bool checkPWMFreq(int pwmFreq){return pwmFreq >= 25 && pwmFreq <= 1500;}

    void timerCallback(const ros::TimerEvent& event)
    {
        if(status == Status::WAITING)
        {

        }
        else if(status == Status::INITING)
        {
            if(!evspin32g4->Open(busDevice))
            {
                ROS_ERROR_ONCE("Failed to initialize EVSPIN32G4: %s. Will retry", evspin32g4->getErrorString().c_str());
                evspin32g4->Close();
                return;
            }

            ROS_INFO("Succussfully initialized EVSPIN32G4 on %s", busDevice.c_str());

            status = Status::RUNNING;
        }
        else if(status == Status::RUNNING)
        {

        }
    }

    void serverCallback(actuator::EVSPIN32G4ActuatorConfig &config, uint32_t level)
    {
        if(level & 0x01)
        {
            if(0 <= config.steer_min_angle &&
                config.steer_min_angle <= config.steer_mid_angle && 
                config.steer_mid_angle <= config.steer_max_angle &&
                config.steer_max_angle <= 180
            )
            {
                steerMinAngle = config.steer_min_angle;
                steerMidAngle = config.steer_mid_angle;
                steerMaxAngle = config.steer_max_angle;
                ROS_INFO("Steer angle are set to min=%d mid=%d max=%d", steerMinAngle, steerMidAngle, steerMaxAngle);
            }
            else
            {
                ROS_ERROR("Invalid steer angle min=%d mid=%d max=%d", 
                    config.steer_min_angle, config.steer_mid_angle, config.steer_max_angle);
            }
        }
        if(level & 0x02)
        {
            if(config.motor_min_rpm <= 0 &&
                config.motor_max_rpm >= 0
            )
            {
                motorMinRPM = config.motor_min_rpm;
                motorMaxRPM = config.motor_max_rpm;
                ROS_INFO("Motor RPM are set to min=%d max=%d", motorMinRPM, motorMaxRPM);
            }
            else
            {
                ROS_ERROR("Invalid motor RPM min=%d max=%d", config.motor_min_rpm, config.motor_max_rpm);
            }
        }
        if(level & 0x04)
        {
            if(status == Status::WAITING)
            {
                busDevice = config.bus_device;
                status = Status::INITING;
            }
            else if(status == Status::INITING)
            {
                busDevice = config.bus_device;
            }
            else if(status == Status::RUNNING)
            {
                std::unique_ptr<EVSPIN32G4> newDevice = std::make_unique<EVSPIN32G4>();
                if(!newDevice->Open(config.bus_device))
                {
                    ROS_ERROR("Invaild i2c bus %s. Using %s", config.bus_device.c_str(),busDevice.c_str());
                }
                else
                {
                    evspin32g4.reset();
                    evspin32g4 = std::move(evspin32g4);
                }
            }
        }
    }
};

int main(int argc, char** argv)
{
    // Initialize the ROS node
    ros::init(argc, argv, "pca9685_actuator_node");

    // Create a NodeHandle
    ros::NodeHandle nh("~");


    EVSPIN32G4Actuator evspin32g4Actuator(nh);
    
    ros::spin();

    return 0;
}
