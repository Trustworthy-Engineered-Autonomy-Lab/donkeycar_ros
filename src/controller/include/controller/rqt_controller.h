#include <rqt_gui_cpp/plugin.h>
#include <ros/ros.h>
#include <QWidget>
#include <QLabel>
#include <QSlider>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QGridLayout>

#include <controller/controller.h>

namespace rqt_controller{

class MainWidget : public QWidget {
    Q_OBJECT

public:
    MainWidget(ros::NodeHandle& nodeHandle, QWidget* parent = nullptr);
private:
    controller::Controller controller;
    float steer;
    float throttle;

    QLabel* steerLabel;
    QLabel* throttleLabel;
};

class RQTController : public rqt_gui_cpp::Plugin 
{
    Q_OBJECT

public:
    RQTController() : rqt_gui_cpp::Plugin()
    {
        setObjectName("DonkeyCarRqtPlugin");
    }
    ~RQTController();
    void initPlugin(qt_gui_cpp::PluginContext& context);
    void shutdownPlugin();

private:
    MainWidget* mainWidget;
    ros::NodeHandle nh;
};

}
