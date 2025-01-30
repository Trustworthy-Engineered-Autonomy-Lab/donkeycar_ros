// #include <linux/input.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <stdio.h>
// #include <stdlib.h>
#include <controller/rqt_controller.h>

#include <string>
#include <memory>

#include <pluginlib/class_list_macros.h>

namespace rqt_controller{

MainWidget::MainWidget(ros::NodeHandle& nodeHandle, QWidget* parent) : QWidget(parent),
        steer(0),throttle(0)
{
    // Create the labels and sliders
    throttleLabel = new QLabel("Throttle Value: 0", this);
    steerLabel = new QLabel("Steer Angle: 0", this);

    QSlider* throttleSlider = new QSlider(Qt::Horizontal, this);
    throttleSlider->setRange(-1000, 1000);  // Set range for slider1 (0 to 100)
    throttleSlider->setValue(0);       // Set initial value

    QSlider* steerSlider = new QSlider(Qt::Horizontal, this);
    steerSlider->setRange(-1000, 1000);  // Set range for slider2 (0 to 100)
    steerSlider->setValue(0);       // Set initial value

    // Connect slider signals to update the labels
    connect(throttleSlider, &QSlider::valueChanged, [this](int value) {
        this->throttle = value/1000.0f;
        this->throttleLabel->setText(QString("Throttle Value: %1").arg(this->throttle));
        this->controller.control(this->throttle, this->steer);
    });

    connect(steerSlider, &QSlider::valueChanged, [this](int value) {
        this->steer = value/1000.0f;
        this->steerLabel->setText(QString("Steer Angle: %1").arg(this->steer));
        this->controller.control(this->throttle, this->steer);
    });

    // Layout for sliders and labels
    QGridLayout* mainLayout = new QGridLayout(this);
    mainLayout->addWidget(throttleLabel,0,0);
    mainLayout->addWidget(steerLabel,1,0);
    mainLayout->addWidget(throttleSlider,0,1,1,2);
    mainLayout->addWidget(steerSlider,1,1,1,2);

    setLayout(mainLayout);
}

RQTController::~RQTController()
{

}

void RQTController::initPlugin(qt_gui_cpp::PluginContext& context) 
{
    nh = ros::NodeHandle("~");
    mainWidget = new MainWidget(nh);

    // Register the widget with the rqt context
    if (context.serialNumber() > 1) {
        mainWidget->setWindowTitle(mainWidget->windowTitle() + " (" + QString::number(context.serialNumber()) + ")");
    }
    context.addWidget(mainWidget);
}

void RQTController::shutdownPlugin() 
{
    // delete mainWidget;
}

}

PLUGINLIB_EXPORT_CLASS(rqt_controller::RQTController, rqt_gui_cpp::Plugin)