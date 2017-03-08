#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "videofile.h"
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <QTimer>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cvimagewidget.h"
#include <QFileDialog>
#include <DPM/dpm.hpp>


using namespace std;
using namespace cv;
using namespace cv::dpm;

VideoFile Camera1;
VideoFile Camera2;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    cout << "Open video files from the cameras" << endl;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::DisplayImages(string FrameNumber)
{


    // Resize the video for displaying to the size of the widget
    WidgetHeight = ui->CVWidget1->height();
    WidgetWidth  = ui->CVWidget1->width();
    ProjWidgetHeight = ui->CVWidgetCenital->height();
    ProjWidgetWidth  = ui->CVWidgetCenital->width();

    cv::resize(Camera1.ActualFrame, Camera1.ActualFrame, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);
    cv::resize(Camera2.ActualFrame, Camera2.ActualFrame, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);
    cv::resize(CenitalPlane, CenitalPlane, {ProjWidgetWidth, ProjWidgetHeight}, INTER_LANCZOS4);

    // Extract Frame number and write it on the frame
    putText(Camera1.ActualFrame, FrameNumber, Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
    putText(Camera2.ActualFrame, FrameNumber, Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

    // Method to display the frame in the CVWidget
    ui->CVWidget1->showImage(Camera1.ActualFrame);
    ui->CVWidget2->showImage(Camera2.ActualFrame);

    // Convert Cenital Plane to 8-bit image
    CenitalPlane.convertTo(CenitalPlane, CV_8UC3, 255.0);
    // Display projected points into Cenital Plane Widget
    ui->CVWidgetCenital->showImage(CenitalPlane);

}

void MainWindow::on_actionOpen_file_triggered()
{
    // Assign cameras a number
    Camera1.CameraNumber = 1;
    Camera2.CameraNumber = 2;

    // Global Path variable should be change if used in other computer
    QString GlobalPath = "/Users/alex/IPCV-MasterThesis/ApplicationCode";

    if (ProgramFlag) {
        // CAMERA 1
        // Get a filename to open
        QString filePath = QFileDialog::getOpenFileName(this, tr("Open VideoFile Camera 1"), GlobalPath, tr("Video Files (*.mpg *.avi *.m4v *.ts *.m2v)"));
        // Convert QString to std::string
        Camera1.InputPath = filePath.toStdString();

        // CAMERA 2
        // Get a filename to open
        filePath = QFileDialog::getOpenFileName(this, tr("Open VideoFile Camera 2"), GlobalPath, tr("Video Files (*.mpg *.avi *.m4v *.ts *.m2v)"));
        // Convert QString to std::string
        Camera2.InputPath = filePath.toStdString();
    }
    else {
        Camera1.InputPath = "/Users/alex/Desktop/Videos Sincronizados/Camera1Sync.m2v";
        Camera2.InputPath = "/Users/alex/Desktop/Videos Sincronizados/Camera2Sync.m2v";
    }

    // Open Video Streams
    Camera1.VideoOpenning(Camera1.InputPath);
    Camera2.VideoOpenning(Camera2.InputPath);

    cout << "Video processing starts" << endl;

    // Create and open the statistics file
    Camera1.VideoStatsFile.open(GlobalPath.toStdString() + "/VideoProcessingStats.txt");
    Camera1.VideoStatsFile << "Frame  Computational Time" << endl;

    // Homography calculation for all the cameras
    Camera1.computeHomography();
    // Homography camera 2
    Camera2.computeHomography();

    // Timer to launch the ProcessVideo() slot
    imageTimer = new QTimer(this);
    connect(imageTimer, SIGNAL(timeout()), this, SLOT(ProcessVideo()));
    imageTimer->start();
}

void MainWindow::ProcessVideo()
{
    // Start the clock for measuring frame consumption
    clock_t begin = clock();

    Camera1.cap >> Camera1.ActualFrame; // Get first video frame
    Camera2.cap >> Camera2.ActualFrame; // Get first video frame

    // Get frame number
    stringstream ss;
    ss << Camera1.cap.get(CAP_PROP_POS_FRAMES);
    string FrameNumber = ss.str().c_str();

    // Check if we achieved the end of the file (e.g. ActualFrame.data is empty)
    if (!Camera1.ActualFrame.data){
        cout << "The processing has finished" << endl;
        Camera1.VideoStatsFile.close();
        imageTimer->blockSignals(true);
        return;
    }

    /* -----------------------*/
    /*      MAIN ALGORITHM    */
    /* -----------------------*/

    // ----------------------- //
    //     BKG SUBSTRACTION    //
    // ----------------------- //

    // Compute Background Mask
    //Camera1.pMOG2->apply(Camera1.ActualFrame, Camera1.BackgroundMask);
    // Improve Background Mask
    //Camera1.maskEnhancement(Camera1.BackgroundMask);

    // ----------------------- //
    //   SEMANTIC PROJECTION   //
    // ----------------------- //
    Camera1.projectSemantic(CenitalPlane);
    Camera2.projectSemantic(CenitalPlane);

    // ----------------------- //
    //     PEOPLE DETECTION    //
    // ----------------------- //

    String CBOption = ui->PeopleDetectorCB->currentText().toStdString();
    if (ui->FastButton->isChecked())
        Camera1.FastRCNNMethod = "fast";
    else if (ui->AccurateButton->isChecked())
        Camera1.FastRCNNMethod = "accurate";

    if (!CBOption.compare("HOG")){
        // HOG Detector
        // Camera 1
        Camera1.HOGPeopleDetection(Camera1.ActualFrame);
        Camera1.paintBoundingBoxes(Camera1.ActualFrame, CBOption, Camera1.HOGBoundingBoxesNMS, Scalar (0, 255, 0), 1);
        Camera1.projectBlobs(Camera1.HOGBoundingBoxesNMS, Camera1.HOGScores, Camera1.Homography, "GREEN", CenitalPlane);

        // Camera 2
        Camera2.HOGPeopleDetection(Camera2.ActualFrame);
        Camera2.paintBoundingBoxes(Camera2.ActualFrame, CBOption, Camera2.HOGBoundingBoxesNMS, Scalar (0, 255, 0), 1);
        Camera2.projectBlobs(Camera2.HOGBoundingBoxesNMS, Camera2.HOGScores, Camera2.Homography, "GREEN", CenitalPlane);
    }
    else if(!CBOption.compare("FastRCNN")){
        // FastRCNN Detector
        Camera1.FastRCNNPeopleDetection(FrameNumber, Camera1.FastRCNNMethod);
        Camera1.paintBoundingBoxes(Camera1.ActualFrame, CBOption, Camera1.RCNNBoundingBoxesNMS, Scalar (0, 0, 255), 1);
        Camera1.projectBlobs(Camera1.RCNNBoundingBoxesNMS, Camera1.RCNNScores, Camera1.Homography, "RED", CenitalPlane);
    }
    else if(!CBOption.compare("DPM")){
        // DPM Detector
        // Camera 1
        Camera1.DPMPeopleDetection(Camera1.ActualFrame);
        Camera1.paintBoundingBoxes(Camera1.ActualFrame, CBOption, Camera1.DPMBoundingBoxes, Scalar (255, 0, 0), 1);
        Camera1.projectBlobs(Camera1.DPMBoundingBoxes, Camera1.DPMScores, Camera1.Homography, "BLUE", CenitalPlane);

        // Camera 2
        Camera2.HOGPeopleDetection(Camera2.ActualFrame);
        Camera2.paintBoundingBoxes(Camera2.ActualFrame, CBOption, Camera2.HOGBoundingBoxesNMS, Scalar (0, 255, 0), 1);
        Camera2.projectBlobs(Camera2.HOGBoundingBoxesNMS, Camera2.HOGScores, Camera2.Homography, "GREEN", CenitalPlane);
    }
    else{
        // HOG Detector
        Camera1.HOGPeopleDetection(Camera1.ActualFrame);
        Camera1.paintBoundingBoxes(Camera1.ActualFrame, CBOption, Camera1.HOGBoundingBoxesNMS, Scalar (0, 255, 0), 1);
        Camera1.projectBlobs(Camera1.HOGBoundingBoxesNMS, Camera1.HOGScores, Camera1.Homography, "GREEN", CenitalPlane);

        // FastRCNN Detector
        Camera1.FastRCNNPeopleDetection(FrameNumber, Camera1.FastRCNNMethod);
        Camera1.paintBoundingBoxes(Camera1.ActualFrame, CBOption, Camera1.RCNNBoundingBoxesNMS, Scalar (0, 0, 255), 1);
        Camera1.projectBlobs(Camera1.RCNNBoundingBoxesNMS, Camera1.RCNNScores, Camera1.Homography, "RED", CenitalPlane);
    }


    // ----------------------- //
    //         DISPLAY         //
    // ----------------------- //
    DisplayImages(FrameNumber);

    // Pause to control the frame rate of the video when the option button is checked
    if (ui->RealTimeButton->isChecked())
        waitKey(30);

    /* -----------------------*/
    /* -----------------------*/

    // Compute the processing time per frame
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    // Save measures to .txt file
    Camera1.VideoStatsFile << FrameNumber << "       " << elapsed_secs << endl;

}

// ---------------------------------------- //
//        HOMOGRAPHY POINTS SELECTION       //
// ---------------------------------------- //

void onMouseCamera1Cenital(int evt, int x, int y, int, void*)
{
    if(evt == CV_EVENT_LBUTTONDOWN) {
        Point pt = Point(x,y);
        cout << "Cenital Frame x = " << pt.x << " y = " << pt.y << endl;
        // Save Points into txt file
        Camera1.PtsDstFile << pt.x << " " << pt.y << endl;
    }
}

void onMouseCamera1Frame(int evt, int x, int y, int, void*)
{
    if(evt == CV_EVENT_LBUTTONDOWN) {
        Point pt = Point(x,y);
        cout << "Camera 1 Frame x = " << pt.x << " y = " << pt.y << endl;
        // Save points into txt
        Camera1.PtsSrcFile << pt.x << " " << pt.y << endl;
    }
}

void MainWindow::on_actionCamera_1_triggered()
{
    // Load the cenital plane
    Mat CenitalFrame = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/CenitalView.png");
    Mat CameraFrame = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/EmptyCamera1.png");

    // Open files
    Camera1.PtsDstFile.open("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/Camera1PtsDstFile.txt");
    Camera1.PtsSrcFile.open("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/Camera1PtsSrcFile.txt");

    String CenitalWindow = "Cenital Frame";
    namedWindow(CenitalWindow);
    setMouseCallback(CenitalWindow, onMouseCamera1Cenital, 0);
    imshow(CenitalWindow, CenitalFrame);

    String FrameWindow = "Camera 1 Frame";
    namedWindow(FrameWindow);
    setMouseCallback(FrameWindow, onMouseCamera1Frame, 0);
    imshow(FrameWindow, CameraFrame);

    if(waitKey()==27) {
        //Camera1.UserSelectedPoints = 1;
        Camera1.PtsDstFile.close();
        Camera1.PtsSrcFile.close();
        destroyWindow(CenitalWindow);
        destroyWindow(FrameWindow);
        return;
    }
}

void onMouseCamera2Cenital(int evt, int x, int y, int, void*)
{
    if(evt == CV_EVENT_LBUTTONDOWN) {
        Point pt = Point(x,y);
        cout << "Cenital Frame x = " << pt.x << " y = " << pt.y << endl;
        // Save Points into txt file
        Camera2.PtsDstFile << pt.x << " " << pt.y << endl;
    }
}

void onMouseCamera2Frame(int evt, int x, int y, int, void*)
{
    if(evt == CV_EVENT_LBUTTONDOWN) {
        Point pt = Point(x,y);
        cout << "Camera 2 Frame x = " << pt.x << " y = " << pt.y << endl;
        // Save points into txt
        Camera2.PtsSrcFile << pt.x << " " << pt.y << endl;
    }
}

void MainWindow::on_actionCamera_2_triggered()
{
    // Load the cenital plane
    Mat CenitalFrame = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/CenitalView.png");
    Mat CameraFrame = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/EmptyCamera2.png");

    // Open files
    Camera2.PtsDstFile.open("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/Camera2PtsDstFile.txt");
    Camera2.PtsSrcFile.open("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/Camera2PtsSrcFile.txt");

    String CenitalWindow = "Cenital Frame";
    namedWindow(CenitalWindow);
    setMouseCallback(CenitalWindow, onMouseCamera2Cenital, 0);
    imshow(CenitalWindow, CenitalFrame);

    String FrameWindow = "Camera 2 Frame";
    namedWindow(FrameWindow);
    setMouseCallback(FrameWindow, onMouseCamera2Frame, 0);
    imshow(FrameWindow, CameraFrame);

    if(waitKey()==27) {
        //Camera1.UserSelectedPoints = 1;
        Camera2.PtsDstFile.close();
        Camera2.PtsSrcFile.close();
        destroyWindow(CenitalWindow);
        destroyWindow(FrameWindow);
        return;
    }
}
