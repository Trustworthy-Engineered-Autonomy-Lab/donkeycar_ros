#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <dynamic_reconfigure/server.h>

#include <actuator/actuator.h>
#include <actuator/PCA9685ActuatorConfig.h>

#include <string>

#include <boost/thread.hpp>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>

class PCA9685
{
    public:
    PCA9685(std::string& deviceName, unsigned freqency = 60, unsigned address = 0x40):
        fileHandle(-1)
    {
        Open(deviceName, address);
    }

    PCA9685()=default;

    ~PCA9685()
    {
        Close();
    }

    bool Open(std::string& deviceName, unsigned freqency = 60, unsigned address = 0x40)
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

        reset();
        return setPWMFreq(freqency);
    }

    void Close()
    {
        if(fileHandle != -1)
        {
            uint8_t Mode = 0;
            readRegister(MODE1, &Mode);
            writeRegister(MODE1, Mode | 0x10);

            close(fileHandle);
            fileHandle = -1;
        }
    }

    bool setPWMFreq(int freqency)
    {
        float prescaleVal = 25000000.0f / (4096 * freqency) - 1;
        uint8_t prescale = static_cast<uint8_t>(std::round(prescaleVal));

        uint8_t Mode = 0;
        if(!readRegister(MODE1, &Mode))
            return false;
        
        if(!(Mode & 0x10))
        {
            uint8_t newMode = (Mode & 0x7F) | 0x10; // Sleep mode
            if(!writeRegister(MODE1, newMode))
                return false;
        }

        if(!writeRegister(PRESCALE, prescale))
            return false;
        if(!writeRegister(MODE1, Mode & 0xEF))
            return false;
        usleep(5000); // Wait for oscillator to stabilize

        if(!readRegister(MODE1, &Mode))
            return false;
        if(!writeRegister(MODE1, Mode | 0x80))
            return false;
        if(!readRegister(MODE1, &Mode))
            return false;
        if (Mode & 0x80)
        {
            errorString = "Failed to restart PCA9685";
            return false;
        }
        return true;
    }

    bool setPWMChannel(int channel, float dutyCycle) 
    {
        uint16_t on = 0;
        uint16_t off = 4096 * dutyCycle;

        if(!writeRegister(LED0_ON_L + 4 * channel, on & 0xFF))
            return false;
        if(!writeRegister(LED0_ON_H + 4 * channel, on >> 8))
            return false;
        if(!writeRegister(LED0_OFF_L + 4 * channel, off & 0xFF))
            return false;
        if(!writeRegister(LED0_OFF_H + 4 * channel, off >> 8))
            return false;
        
        return true;
    }

    const std::string& getErrorString()
    {
        return errorString;
    }
    

    private:

    int fileHandle;
    std::string errorString;

    static constexpr uint8_t MODE1 = 0x00;
    static constexpr uint8_t MODE2 = 0x01;
    static constexpr uint8_t LED0_ON_L = 0x06;
    static constexpr uint8_t LED0_ON_H = 0x07;
    static constexpr uint8_t LED0_OFF_L = 0x08;
    static constexpr uint8_t LED0_OFF_H = 0x09;
    static constexpr uint8_t PRESCALE = 0xFE;

    void reset()
    {
        uint8_t buffer[2] = {0x00, 0x06};
        write(fileHandle, buffer, 2);
    }

    bool writeRegister(uint8_t reg, uint8_t value) 
    {
        uint8_t buffer[2] = {reg, value};

        if (write(fileHandle, buffer, 2) != 2) 
        {
            errorString = "Failed to write register " + std::to_string(reg);
            return false;
        }
        return true;
    }

    bool readRegister(uint8_t reg, uint8_t* value) 
    {
        if (write(fileHandle, &reg, 1) != 1) 
        {
            errorString = "Failed to read register " + std::to_string(reg);
            return false;
        }

        if (read(fileHandle, value, 1) != 1) 
        {
            errorString = "Failed to read register " + std::to_string(reg);
            return false;
        }
        return true;
    }
};

class PCA9685Actuator: public actuator::Actuator
{
    public:
    PCA9685Actuator(ros::NodeHandle& nodeHandle):server(nodeHandle),
        nh(nodeHandle),
        status(Status::WAITING),
        pca9685{std::make_unique<PCA9685>()}
    {

        throttleChannel = 0;
        steerChannel = 1;
        
        steerMinPW = 1000;
        steerMaxPW = 2000;
        steerMidPW = 1500;

        throttleMinPW = 1000;
        throttleMaxPW = 2000;
        throttleMidPW = 1500;

        server.setCallback(boost::bind(&PCA9685Actuator::serverCallback,this,boost::placeholders::_1,boost::placeholders::_2));
        timer = nodeHandle.createTimer(ros::Duration(1), boost::bind(&PCA9685Actuator::timerCallback, this, boost::placeholders::_1));
    }

    void actuate(float throttle, float steer)
    {
        if(status == Status::RUNNING)
        {
            float steerPW;
            if(steer > 0)
            {
                steerPW = steer * (steerMaxPW - steerMidPW) + steerMidPW;
            }
            else
            {
                steerPW = steer * (steerMidPW - steerMinPW) + steerMidPW;
            }

            float throttlePW;
            if(throttle > 0)
            {
                throttlePW = throttle * static_cast<float>(throttleMaxPW - throttleMidPW) + throttleMidPW;
            }
            else
            {
                throttlePW = throttle * static_cast<float>(throttleMidPW - throttleMinPW) + throttleMidPW;
            }

            ROS_DEBUG("Throttle pwm pulse width: %f us, Steer pwm pulse width: %f us", throttlePW, steerPW);

            float duration = 1000000/(float)pwmFreq;

            float throttleDuty = throttlePW/duration;
            float steerDuty = steerPW/duration;

            if(!pca9685->setPWMChannel(throttleChannel, throttleDuty))
            {
                ROS_ERROR("Failed to set throttle value: %s. Will retry", pca9685->getErrorString().c_str());
                status = Status::INITING;
                pca9685->Close();
                return;
            }
            
            if(!pca9685->setPWMChannel(steerChannel, steerDuty))
            {
                ROS_ERROR("Failed to set steer angle: %s. Will retry", pca9685->getErrorString().c_str());
                status = Status::INITING;
                pca9685->Close();
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

    std::unique_ptr<PCA9685> pca9685;
    std::string busDevice;
    int pwmFreq;

    int throttleChannel;
    int steerChannel;

    int throttleMinPW;
    int throttleMaxPW;
    int throttleMidPW;

    int steerMinPW;
    int steerMaxPW;
    int steerMidPW;

    ros::Timer timer;
    Status status;

    ros::NodeHandle& nh;

    dynamic_reconfigure::Server<actuator::PCA9685ActuatorConfig> server;

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
            if(!pca9685->Open(busDevice, pwmFreq))
            {
                ROS_ERROR_ONCE("Failed to initialize PCA9685: %s. Will retry", pca9685->getErrorString().c_str());
                pca9685->Close();
                return;
            }

            ROS_INFO("Succussfully initialized PCA9685 on %s", busDevice.c_str());

            // ROS_INFO("-----------------------------------------------");
            // ROS_INFO(" PCA9685 Configuration");
            // ROS_INFO("-----------------------------------------------");
            // ROS_INFO(" PWM freqency %dHz", pwmFreq);
            // ROS_INFO("-----------------------------------------------");
            // ROS_INFO("%-20s | %-10s | %-10s", "Parameter", "Throttle", "Steer");
            // ROS_INFO("---------------------+------------+------------");
            // ROS_INFO("%-20s | %-10d | %-10d", "PWM channel", throttleChannel, steerChannel);
            // ROS_INFO("%-20s | %-10d | %-10d", "Min pulse width (us)", throttleMinPW, steerMinPW);
            // ROS_INFO("%-20s | %-10d | %-10d", "Mid pulse width (us)", throttleMidPW, steerMidPW);
            // ROS_INFO("%-20s | %-10d | %-10d", "Max pulse width (us)", throttleMaxPW, steerMaxPW);
            // ROS_INFO("-----------------------------------------------");

            status = Status::RUNNING;
        }
        else if(status == Status::RUNNING)
        {

        }
    }

    void serverCallback(actuator::PCA9685ActuatorConfig &config, uint32_t level)
    {
        if(level & 0x01)
        {
            if(checkPWMPW(config.steer_min_pulsewidth, config.steer_mid_pulsewidth, config.steer_max_pulsewidth))
            {
                steerMinPW = config.steer_min_pulsewidth;
                steerMidPW = config.steer_mid_pulsewidth;
                steerMaxPW = config.steer_max_pulsewidth;
                ROS_INFO("PWM pulse width for steer is set to %dus %dus %dus", steerMinPW, steerMidPW, steerMaxPW);
            }
            else
            {
                ROS_WARN("Invalid pwm pulse width %dus %dus %dus for steer. Using %dus %dus %dus", 
                    config.steer_min_pulsewidth, 
                    config.steer_mid_pulsewidth, 
                    config.steer_max_pulsewidth,
                    steerMinPW, steerMidPW, steerMaxPW);
            }
        }
        if(level & 0x02)
        {
            if(checkPWMPW(config.throttle_min_pulsewidth, config.throttle_mid_pulsewidth, config.throttle_max_pulsewidth))
            {
                throttleMinPW = config.throttle_min_pulsewidth;
                throttleMidPW = config.throttle_mid_pulsewidth;
                throttleMaxPW = config.throttle_max_pulsewidth;
                ROS_INFO("PWM pulse width for throttle is set to %dus %dus %dus", throttleMinPW, throttleMidPW, throttleMaxPW);
            }
            else
            {
                ROS_WARN("Invalid pulse width %dus %dus %dus for throttle. Using %dus %dus %dus",
                    config.throttle_min_pulsewidth,
                    config.throttle_mid_pulsewidth,
                    config.throttle_max_pulsewidth,
                    throttleMinPW, throttleMidPW, throttleMaxPW);
            }
        }
        if(level & 0x04)
        {
            if(!checkPWMChannel(config.steer_pwm_channel))
            {
                ROS_WARN("Invalid PCA9685 channel number %d for steer. Using %d", config.steer_pwm_channel, steerChannel);
                return;
            }
            if(!checkPWMChannel(config.throttle_pwm_channel))
            {
                ROS_WARN("Invalid PCA9685 channel number %d for throttle. Using %d", config.throttle_pwm_channel, throttleChannel);
                return;
            }
            if(config.steer_pwm_channel == config.throttle_pwm_channel)
            {
                ROS_WARN("Channel number for steer and throttle can not be equal");
                return;
            }
            
            if(steerChannel != config.steer_pwm_channel)
            {
                if(status == Status::RUNNING)
                {
                    pca9685->setPWMChannel(steerChannel,0);
                }

                steerChannel = config.steer_pwm_channel;
            }

            if(throttleChannel != config.throttle_pwm_channel)
            {
                if(status == Status::RUNNING)
                {
                    pca9685->setPWMChannel(throttleChannel,0);
                }

                throttleChannel = config.throttle_pwm_channel;
            }
            ROS_INFO("PCA9865 channel numbers is set to steer: %d throttle: %d", steerChannel, throttleChannel);
        }
        if(level & 0x08)
        {
            if(!checkPWMFreq(config.pwm_frequency))
            {
                ROS_WARN("Invaild PCA9685 pwm frequency %d. Using %d", config.pwm_frequency, pwmFreq);
            }
            else
            {
                pwmFreq = config.pwm_frequency;
                if(status == Status::RUNNING)
                {
                    pca9685->setPWMFreq(pwmFreq);
                }
                
                ROS_INFO("PCA9685 pwm frequency is set to %d", pwmFreq);
            }
        }
        if(level & 0x10)
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
                std::unique_ptr<PCA9685> newPCA9685 = std::make_unique<PCA9685>();
                if(!newPCA9685->Open(config.bus_device, pwmFreq))
                {
                    ROS_ERROR("Invaild i2c bus %s. Using %s", config.bus_device.c_str(),busDevice.c_str());
                }
                else
                {
                    pca9685.reset();
                    pca9685 = std::move(newPCA9685);
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


    PCA9685Actuator pca9685Actuator(nh);
    
    ros::spin();

    return 0;
}
