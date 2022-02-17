#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_YoloV5_Window.h"
#include <QtWidgets>
#include <QString>

class YoloV5_Window : public QMainWindow
{
    Q_OBJECT

public:
    YoloV5_Window(QWidget *parent = Q_NULLPTR);

private:
    Ui::YoloV5_WindowClass ui;

private:
    QPushButton* btnOK;
    QPushButton* btnSelectFile;
    
    QString selectImgName;
    QLabel* labelDisplay;

private slots:
    void clickOK();
    void clickSelectFile();

};
