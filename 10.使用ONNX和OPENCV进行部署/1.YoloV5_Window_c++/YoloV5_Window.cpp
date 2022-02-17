

#include "YoloV5_Window.h"
#include "yolo.h"

Net_config yolo_nets[4] = {
	{0.5, 0.5, 0.5, "yolov5s"},
	{0.5, 0.5, 0.5,  "yolov5m"},
	{0.5, 0.5, 0.5, "yolov5l"},
	{0.5, 0.5, 0.5, "yolov5x"}
};

YoloV5_Window::YoloV5_Window(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
	
	ui.mainToolBar->hide();
	ui.menuBar->hide();
	ui.statusBar->hide();

	QGridLayout* windowLayout = new QGridLayout(this);
	btnSelectFile = new QPushButton();
	btnSelectFile->setText(QString::fromLocal8Bit("选择图片文件："));
	btnOK = new QPushButton();
	btnOK->setText(QString::fromLocal8Bit("确定"));
	labelDisplay = new QLabel();
	

	windowLayout->addWidget(btnSelectFile, 0, 0,1,2);
	windowLayout->addWidget(btnOK, 0, 2,1,2);
	windowLayout->addWidget(labelDisplay, 1, 0, 4, 4);

	this->centralWidget()->setLayout(windowLayout);
	

	connect(btnSelectFile, &QPushButton::clicked, this, &YoloV5_Window::clickSelectFile);
	connect(btnOK, &QPushButton::clicked, this, &YoloV5_Window::clickOK);

}

void YoloV5_Window::clickSelectFile() {
	QFileDialog* fileDialog;
	fileDialog = new QFileDialog(this);
	fileDialog->setNameFilter(tr("Images(*.jpg *.png)"));
	if (fileDialog->exec())
	{
		selectImgName = fileDialog->selectedFiles()[0];
	}

}

void YoloV5_Window::clickOK() {

	YOLO yolo_model(yolo_nets[0]);

	string imgpath = selectImgName.toStdString();
	Mat srcimg = imread(imgpath);
	yolo_model.detect(srcimg);

	/*static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();*/

	QString newimg;
	newimg = QString::fromStdString(imgpath);
	QString dealpath = newimg.split(".")[0] + "reg." + newimg.split(".")[1];
	imwrite(dealpath.toStdString(), srcimg);
	QPixmap pixmap(dealpath);
	labelDisplay->setPixmap(pixmap);
	//labelDisplay->show();
}
