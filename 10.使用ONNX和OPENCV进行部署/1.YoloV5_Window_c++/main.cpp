#include "YoloV5_Window.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    YoloV5_Window w;
    w.setWindowTitle(QString::fromLocal8Bit("yolov5s π”√GUI"));
    w.show();
    return a.exec();
}
