#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <fstream>
#include <iostream>
#include <string>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cvimagewidget.h"


using namespace std;
using namespace cv;


VideoCapture cap;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // GLOBAL PATH
    string GlobalPath = "/Users/alex/IPCV-MasterThesis/ApplicationCode";
    // VIDEO
    string inputvideo   = GlobalPath + "/Inputs/Vid1.mpg";


    // Open the videofile to check if it exists
    cap.open(inputvideo);
    if (!cap.isOpened()) {
        cout << "Could not open video file " << inputvideo << endl;
        return;
    }

    // Timer to launch the Process Video slot
    imageTimer = new QTimer(this);
    connect(imageTimer, SIGNAL(timeout()), this, SLOT(ProcessVideo()));
    imageTimer->start();

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::ProcessVideo(){

    Mat ActualFrame;    
    cap >> ActualFrame; // Get first video frame

    // Check if we achieved the end of the file (e.g. ActualFrame.data is empty)
    if (!ActualFrame.data){
        cout << "The processing has finished" << endl;
        return;
    }


        /* MAIN ALGORITHM*/
    /* -----------------------*/


    /* -----------------------*/


    int Height = ui->CVWidget->height();
    int Width  = ui->CVWidget->width();

    cv::resize(ActualFrame, ActualFrame, {Height, Width} );

    // Extract Frame number
    stringstream ss;
    ss << cap.get(CAP_PROP_POS_FRAMES);
    putText(ActualFrame, ss.str().c_str(), cv::Point(15, 15),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255)); //text in red
    cout << endl << "Processing frame " << ss.str() << endl;


    ui->CVWidget->showImage(ActualFrame);
    // Pause to control the frame rate of the video
    //waitKey(30);

}
