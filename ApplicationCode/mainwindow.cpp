#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "camerastream.h"
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
#include "peopledetector.h"


using namespace std;
using namespace cv;
using namespace cv::dpm;

CameraStream Camera1;
CameraStream Camera2;
CameraStream Camera3;
PeopleDetector PeopleDetec;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->textBrowser->append("Open video files from the cameras");
}

MainWindow::~MainWindow()
{
    delete [] Camera1.ArrayProjectedPoints;
    delete [] Camera2.ArrayProjectedPoints;
    delete [] Camera3.ArrayProjectedPoints;
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
    cv::resize(Camera3.ActualFrame, Camera3.ActualFrame, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);
    cv::resize(CenitalPlane, CenitalPlane, {ProjWidgetWidth, ProjWidgetHeight}, INTER_LANCZOS4);

    // Extract Frame number and write it on the frame
    putText(Camera1.ActualFrame, FrameNumber, Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
    putText(Camera2.ActualFrame, FrameNumber, Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
    putText(Camera3.ActualFrame, FrameNumber, Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

    // Display images into UI CVWidgets
    ui->CVWidget1->showImage(Camera1.ActualFrame);
    ui->CVWidget2->showImage(Camera2.ActualFrame);
    ui->CVWidget3->showImage(Camera3.ActualFrame);

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
    Camera3.CameraNumber = 3;

    // Global Path variable should be change if used in other computer
    QString GlobalPath = "/Users/alex/IPCV-MasterThesis/ApplicationCode";

    if (ProgramFlag) {
        // Opens the QFileDialog
        QStringList filenames = QFileDialog::getOpenFileNames(this, tr("Open video files for the cameras"), GlobalPath, tr("Video Files (*.mpg *.avi *.m4v *.ts *.m2v)"));

        // Save cameras paths
        Camera1.InputPath = filenames.at(Camera1.CameraNumber - 1).toStdString();
        Camera2.InputPath = filenames.at(Camera2.CameraNumber - 1).toStdString();
        Camera3.InputPath = filenames.at(Camera3.CameraNumber - 1).toStdString();
    }
    else {
        Camera1.InputPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 1/Videos/Camera1Sync.m2v";
        Camera2.InputPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 1/Videos/Camera2Sync.m2v";
        Camera3.InputPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 1/Videos/Camera3Sync.m2v";
    }

    // Open Video Streams
    Camera1.VideoOpenning(Camera1.InputPath);
    Camera2.VideoOpenning(Camera2.InputPath);
    Camera3.VideoOpenning(Camera3.InputPath);

    ui->textBrowser->append("Videos correctly openned");

    // Create and open the statistics file
    Camera1.VideoStatsFile.open(GlobalPath.toStdString() + "/VideoProcessingStats.txt");
    Camera1.VideoStatsFile << "Frame  Computational Time" << endl;

    // Homography calculation for all the cameras
    Camera1.computeHomography();
    Camera2.computeHomography();
    Camera3.computeHomography();

    // Project Floor Points
    Camera1.ProjectFloorPoints();
    Camera2.ProjectFloorPoints();
    Camera3.ProjectFloorPoints();

    ui->textBrowser->append("Homographies for cameras correctly calculated");
    ui->textBrowser->append("Processing starts");

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
    Camera3.cap >> Camera3.ActualFrame; // Get first video frame

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

    // ---------------------------------------- //
    //     BKG SUBSTRACTION & BLOB EXTRACTION   //
    // ---------------------------------------- //

    // CAMERA 1
    // Compute Background Mask
    Camera1.pMOG2->apply(Camera1.ActualFrame, Camera1.BackgroundMask);
    // Improve Background Mask
    Camera1.maskEnhancement(Camera1.BackgroundMask);
    Camera1.extractFGBlobs(Camera1.BackgroundMask);
    Camera1.ExtractFGImages(Camera1.ActualFrame, Camera1.FGBlobs);

    // CAMERA 2
    // Compute Background Mask
    Camera2.pMOG2->apply(Camera2.ActualFrame, Camera2.BackgroundMask);
    // Improve Background Mask
    Camera2.maskEnhancement(Camera2.BackgroundMask);
    Camera2.extractFGBlobs(Camera2.BackgroundMask);
    Camera2.ExtractFGImages(Camera2.ActualFrame, Camera2.FGBlobs);

    // CAMERA 3
    // Compute Background Mask
    Camera3.pMOG2->apply(Camera3.ActualFrame, Camera3.BackgroundMask);
    // Improve Background Mask
    Camera3.maskEnhancement(Camera3.BackgroundMask);
    Camera3.extractFGBlobs(Camera3.BackgroundMask);
    Camera3.ExtractFGImages(Camera3.ActualFrame, Camera3.FGBlobs);

    cout << "Camera1 -> BKG 1(empty) 0(something): " << Camera1.EmptyBackground << endl;
    cout << "Camera2 -> BKG 1(empty) 0(something): " << Camera2.EmptyBackground << endl;
    cout << "Camera3 -> BKG 1(empty) 0(something): " << Camera3.EmptyBackground << endl;

    //imshow("Camera 1 BS", Camera1.BackgroundMask);
    //imshow("Camera 2 BS", Camera2.BackgroundMask);
    //imshow("Camera 3 BS", Camera3.BackgroundMask);

    // ----------------------- //
    //   SEMANTIC PROJECTION   //
    // ----------------------- //

    Camera1.drawSemantic(CenitalPlane);
    Camera2.drawSemantic(CenitalPlane);
    Camera3.drawSemantic(CenitalPlane);

    if (atoi(FrameNumber.c_str()) == 1) {
        Camera1.saveWarpImages(Camera1.ActualFrame, Camera1.Homography, FrameNumber);
        Camera2.saveWarpImages(Camera2.ActualFrame, Camera2.Homography, FrameNumber);
        Camera3.saveWarpImages(Camera3.ActualFrame, Camera3.Homography, FrameNumber);
    }

    // ------------------------------------------- //
    //     PEOPLE DETECTION & BLOBS PROJECTION     //
    // ------------------------------------------- //

    // Main Method
    String CBOption = ui->PeopleDetectorCB->currentText().toStdString();
    // Representation Method
    String RepresentationOption = ui->RepresentationCB->currentText().toStdString();
    // Using or not Foreground filtering for PD
    bool PDFilteringOption = ui->PDFiltering->isChecked();

    // FastRCNN Method
    if (ui->FastButton->isChecked())
        Camera1.FastRCNNMethod = "fast";
    else if (ui->AccurateButton->isChecked())
        Camera1.FastRCNNMethod = "accurate";

    if (FlagText)
        ui->textBrowser->append(QString::fromStdString(CBOption) + " Detector in use");

    PeopleDetec.MainPeopleDetection(Camera1, Camera2, Camera3, CBOption, RepresentationOption, PDFilteringOption, CenitalPlane);


    // ----------------------- //
    //         DISPLAY         //
    // ----------------------- //

    DisplayImages(FrameNumber);

    /* -----------------------*/
    /* -----------------------*/

    // Compute the processing time per frame
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    // Save measures to .txt file
    Camera1.VideoStatsFile << FrameNumber << "       " << elapsed_secs << endl;

    // Turn off message display
    FlagText = 0;
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
    Mat CenitalFrame = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/CenitalViewRombo.png");
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
    Mat CenitalFrame = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/CenitalViewRombo.png");
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
        Camera2.PtsDstFile.close();
        Camera2.PtsSrcFile.close();
        destroyWindow(CenitalWindow);
        destroyWindow(FrameWindow);
        return;
    }
}

void onMouseCamera3Cenital(int evt, int x, int y, int, void*)
{
    if(evt == CV_EVENT_LBUTTONDOWN) {
        Point pt = Point(x,y);
        cout << "Cenital Frame x = " << pt.x << " y = " << pt.y << endl;
        // Save Points into txt file
        Camera3.PtsDstFile << pt.x << " " << pt.y << endl;
    }
}

void onMouseCamera3Frame(int evt, int x, int y, int, void*)
{
    if(evt == CV_EVENT_LBUTTONDOWN) {
        Point pt = Point(x,y);
        cout << "Camera 3 Frame x = " << pt.x << " y = " << pt.y << endl;
        // Save points into txt
        Camera3.PtsSrcFile << pt.x << " " << pt.y << endl;
    }
}

void MainWindow::on_actionCamera_3_triggered()
{
    // Load the cenital plane
    Mat CenitalFrame = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/CenitalViewRombo.png");
    Mat CameraFrame = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/EmptyCamera3.png");

    // Open files
    Camera3.PtsDstFile.open("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/Camera3PtsDstFile.txt");
    Camera3.PtsSrcFile.open("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/Camera3PtsSrcFile.txt");

    String CenitalWindow = "Cenital Frame";
    namedWindow(CenitalWindow);
    setMouseCallback(CenitalWindow, onMouseCamera3Cenital, 0);
    imshow(CenitalWindow, CenitalFrame);

    String FrameWindow = "Camera 3 Frame";
    namedWindow(FrameWindow);
    setMouseCallback(FrameWindow, onMouseCamera3Frame, 0);
    imshow(FrameWindow, CameraFrame);

    if(waitKey()==27) {
        Camera3.PtsDstFile.close();
        Camera3.PtsSrcFile.close();
        destroyWindow(CenitalWindow);
        destroyWindow(FrameWindow);
        return;
    }
}

void MainWindow::on_PauseCheckBox_clicked(bool checked)
{
    if (checked) {
        ui->textBrowser->append("Paused");
        imageTimer->blockSignals(true);
    }
    else {
        ui->textBrowser->append("Resumed");
        imageTimer->blockSignals(false);
    }
}
