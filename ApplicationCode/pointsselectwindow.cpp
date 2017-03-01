#include "pointsselectwindow.h"
#include "ui_pointsselectwindow.h"
#include "cvimagewidget.h"
#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void onMouse3(int evt, int x, int y, int, void*);

pointsselectwindow::pointsselectwindow(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::pointsselectwindow)
{
    ui->setupUi(this);

    PointsWidgetHeight = ui->CVWidgetPoints->height();
    PointsWidgetWidth  = ui->CVWidgetPoints->width();

    // Load the cenital plane
    Mat CenitalFrame = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/CenitalView.png");
    cv::resize(CenitalFrame, CenitalFrame, {PointsWidgetWidth, PointsWidgetHeight}, INTER_LANCZOS4);

    ui->CVWidgetPoints->showImage(CenitalFrame);
}

pointsselectwindow::~pointsselectwindow()
{
    delete ui;
}

