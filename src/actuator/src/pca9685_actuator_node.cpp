#include <ros/ros.h>
#include <sensor_msgs/Joy.h>

#include <actuator/actuator.h>

#include <string>

#include <boost/thread.hpp>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>

class PCA9685
{
    public:
    PCA9685(std::string& deviceName, unsigned freqency = 60, unsigned address = 0x40)
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

    void Close()
    {
        uint8_t Mode = 0;
        readRegister(MODE1, &Mode);
        writeRegister(MODE1, Mode | 0x10);

        close(fileHandle);
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
    PCA9685Actuator(ros::NodeHandle& nodeHandle):actuator::Actuator(nodeHandle),
        nh(nodeHandle),
        status(Status::INITING)
    {

        throttleChannel = nodeHandle.param<int>("throttle_pwm_channel",0);
        steerChannel = nodeHandle.param<int>("steer_pwm_channel",1);
        
        steerMinPW = nodeHandle.param<int>("steer_min_pulsewidth", 1000);
        steerMaxPW = nodeHandle.param<int>("steer_max_pulsewidth", 2000);
        steerMidPW = nodeHandle.param<int>("steer_mid_pulsewidth", 1500);

        throttleMinPW = nodeHandle.param<int>("throttle_min_pulsewidth", 1000);
        throttleMaxPW = nodeHandle.param<int>("throttle_max_pulsewidth", 2000);
        throttleMidPW = nodeHandle.param<int>("throttle_mid_pulsewidth", 1500);

        if (throttleChannel < 0 || throttleChannel > 15)
        {
            ROS_WARN("Invalid PCA9685 channel number %d for throttle! Using default values 0", throttleChannel);
            throttleChannel = 0;
        }

        if (steerChannel < 0 || steerChannel > 15)
        {
            ROS_WARN("Invalid PCA9685 channel number %d for steer! Using default values 1", steerChannel);
            steerChannel = 1;
        }

        if(steerMinPW < 0 || steerMaxPW < steerMinPW || steerMidPW > steerMaxPW || steerMidPW < steerMinPW)
        {
            ROS_WARN("Invalid pulse width %dus %dus %dus for steer! Using default values 1000us 1500us 2000us", steerMinPW, steerMidPW, steerMaxPW);
            steerMinPW = 1000;
            steerMaxPW = 2000;
            steerMidPW = 1500;
        }

        if(throttleMinPW < 0 || throttleMaxPW < throttleMinPW || throttleMidPW > throttleMaxPW || throttleMidPW < throttleMinPW)
        {
            ROS_WARN("Invalid pulse width %dus %dus %dus for throttle! Using default values 1000us 1500us 2000us", throttleMinPW, throttleMidPW, throttleMaxPW);
            throttleMinPW = 1000;
            throttleMaxPW = 2000;
            throttleMidPW = 1500;
        }
        ROS_INFO("-----------------------------------------------");
        ROS_INFO(" PCA9685 Configuration");
        ROS_INFO("-----------------------------------------------");
        ROS_INFO(" PWM freqency %dHz", pwmFreq);
        ROS_INFO("-----------------------------------------------");
        ROS_INFO("%-20s | %-10s | %-10s", "Parameter", "Throttle", "Steer");
        ROS_INFO("---------------------+------------+------------");
        ROS_INFO("%-20s | %-10d | %-10d", "PWM channel", throttleChannel, steerChannel);
        ROS_INFO("%-20s | %-10d | %-10d", "Min pulse width (us)", throttleMinPW, steerMinPW);
        ROS_INFO("%-20s | %-10d | %-10d", "Mid pulse width (us)", throttleMidPW, steerMidPW);
        ROS_INFO("%-20s | %-10d | %-10d", "Max pulse width (us)", throttleMaxPW, steerMaxPW);
        ROS_INFO("-----------------------------------------------");

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
                throttlePW = throttle * (throttleMaxPW - throttleMidPW) + steerMidPW;
            }
            else
            {
                throttlePW = throttle * (throttleMidPW - throttleMinPW) + throttleMidPW;
            }

            ROS_DEBUG("Throttle pwm pulse width: %f us, Steer pwm pulse width: %f us", throttlePW, steerPW);

            float duration = 1000000/(float)pwmFreq;

            float throttleDuty = throttlePW/duration;
            float steerDuty = steerPW/duration;

            if(!pca9685.setPWMChannel(throttleChannel, throttleDuty))
            {
                ROS_ERROR("Failed to set throttle value: %s. Will retry", pca9685.getErrorString().c_str());
                pca9685.Close();
                status = Status::INITING;
                return;
            }
            
            if(!pca9685.setPWMChannel(steerChannel, steerDuty))
            {
                ROS_ERROR("Failed to set steer angle: %s. Will retry", pca9685.getErrorString().c_str());
                pca9685.Close();
                status = Status::INITING;
                return;
            }
        }
    }

    private:

    enum class Status
    {
        INITING,
        RUNNING,
    };

    PCA9685 pca9685;
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

    void timerCallback(const ros::TimerEvent& event)
    {
        if(status == Status::INITING)
        {
            int busNumber = nh.param<int>("bus_number", 1);
    
            std::string busName = "/dev/i2c-" + std::to_string(busNumber);

            pwmFreq = nh.param<int>("pwm_frequency", 60);

            if (pwmFreq < 25 || pwmFreq > 1500)
            {
                ROS_WARN_ONCE("Invalid PCA9685 pwm frequency %dHz! Using default values 60 Hz", pwmFreq);
                pwmFreq = 60;
            }

            if(!pca9685.Open(busName, pwmFreq))
            {
                pca9685.Close();
                ROS_ERROR_ONCE("Failed to initialize PCA9685: %s. Will retry", pca9685.getErrorString().c_str());
                return;
            }

            status = Status::RUNNING;
        }
        else if(status == Status::RUNNING)
        {

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
