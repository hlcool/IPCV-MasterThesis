#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <fstream>
#include <iostream>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Load an image
    Mat image = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/aux.png", true);

    int Height = ui->CVWidget->height();
    int Width  = ui->CVWidget->width();

    cout << "The size of the CVWidget is:" << endl;
    cout << "Height: " << Height << endl;
    cout << "Width: " << Width << endl;

    cv::resize(image, image, {Height, Width} );

    ui->CVWidget->showImage(image);
}

MainWindow::~MainWindow()
{
    delete ui;
}

