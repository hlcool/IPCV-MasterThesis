#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "barrier.h"
#include "camerastream.h"
#include "cameraworker.h"
#include "aboutdialog.h"
#include <QDebug>
#include <QLabel>
#include <string>
#include <QThread>
#include <QMutex>
#include <iostream>
#include <QGridLayout>
#include <QFileDialog>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

vector<CameraStream> Cameras;
ofstream PtsDstFile;
ofstream PtsSrcFile;
QMutex mutexWindow;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->textBrowser->append("Interface thread started");
    ui->textBrowser->append("Press File and Open video files");

    // MetaType register for connection between signal and slots
    qRegisterMetaType<Mat>("Mat");
    qRegisterMetaType<String>("String");
    qRegisterMetaType<CameraStream>("CameraStream");
    qDebug() << "Main thread " << QThread::currentThreadId();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionOpen_file_triggered()
{
    // Global Path variable should be change if used in other computer
    GlobalPath = "/Users/alex/IPCV-MasterThesis/ApplicationCode";

    if (ProgramFlag) {
        // Opens the QFileDialog to get files
        filenames = QFileDialog::getOpenFileNames(this, tr("Open video files for the cameras"), GlobalPath, tr("Video Files (*.mpg *.avi *.m4v *.ts *.m2v)"));
    }
    else {
        filenames << "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Videos/Camera1Sync.m2v";
        filenames << "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Videos/Camera2Sync.m2v";
        filenames << "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Videos/Camera3Sync.m2v";
    }
    threadStarting();
}
void MainWindow::threadStarting()
{
    // Number of cameras that will be used
    numCams = 3;

    // Barrier point so the threads have a point to meet before displaying the frame.
    // Same barrier for all the threads
    Barrier barrier(numCams);

    // Thread creating loop
    for (int i = 0; i < numCams; i++){
        // Create threads
        threads[i] = new QThread;

        // Create the camera class for workers
        CameraStream Camera;
        // Fill some camera start variables
        fillCameraVariables(Camera, i);

        // New CameraWorker initialize with Camera(CameraStream) and the barrier
        CameraWorkers[i] = new CameraWorker(Camera, barrier);

        // Fill UI widget size in CameraWorkers
        CameraWorkers[i]->WidgetHeight = ui->CVWidget1->height();
        CameraWorkers[i]->WidgetWidth = ui->CVWidget1->width();
        CameraWorkers[i]->CBOption = ui->PeopleDetectorCB->currentText().toStdString();

        // Move CameraWorker to thread
        CameraWorkers[i]->moveToThread(threads[i]);

        // Connect signals to slot between thread and CameraWorkers
        connectSignals2Slots(threads[i], CameraWorkers[i]);

        // Thread is started
        threads[i]->start();
        ui->textBrowser->append(QString::fromStdString("Thread from camera " + to_string(i+1) + " started"));
    }
}

void MainWindow::connectSignals2Slots(QThread *thread, CameraWorker *worker)
{
    // THREAD SIGNAL CONECTION
    // Thread starting with processVideo slot
    connect(thread, SIGNAL(started()), worker, SLOT(preProcessVideo()));
    // Thread finished with delete slot
    connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));

    // WORKER SIGNAL CONNECTIONS
    // cenitalJoined signal with displayFrame slot
    connect(this, SIGNAL(cenitalJoined(Mat, Mat, int)), this, SLOT(displayFrame(Mat, Mat, int)));
    // frameFinished signal with updateVariables and joinCenitalFrames slots
    connect(worker, SIGNAL(frameFinished(Mat, Mat, int)), this, SLOT(updateVariables(Mat, Mat, int)));
    connect(worker, SIGNAL(frameFinished(Mat, Mat, int)), this, SLOT(joinCenitalFrames(Mat, Mat, int)));
    connect(worker, SIGNAL(PedestrianDetectionFinished(int)), this, SLOT(sharePedestrianDetections(int)));

    // finished signal with quit and deleteLater slots
    connect(worker, SIGNAL(finished()), thread, SLOT(quit()));
    connect(worker, SIGNAL(finished()), worker, SLOT(deleteLater()));
}

void MainWindow::fillCameraVariables(CameraStream &Camera, int i)
{
    // Camera Number
    Camera.CameraNumber = i + 1;
    // Global path
    Camera.GlobalPath = GlobalPath.toStdString();
    // Input Path
    Camera.InputPath = filenames.at(Camera.CameraNumber - 1).toStdString();
}

void MainWindow::updateVariables(Mat Frame, Mat CenitalPlane, int CameraNumber)
{
    if (CameraNumber == 1){
        // Update messages only for one camera
        if(CameraWorkers[CameraNumber-1]->CBOption.compare(ui->PeopleDetectorCB->currentText().toStdString()))
            ui->textBrowser->append(ui->PeopleDetectorCB->currentText() + " People Detector in use");
    }

    // Widget size variables
    CameraWorkers[CameraNumber-1]->WidgetWidth  = ui->CVWidget1->width();
    CameraWorkers[CameraNumber-1]->WidgetHeight = ui->CVWidget1->height();

    // People detection options
    CameraWorkers[CameraNumber-1]->CBOption = ui->PeopleDetectorCB->currentText().toStdString();
    if(!CameraWorkers[CameraNumber-1]->CBOption.compare("Semantic Detector"))
        CameraWorkers[CameraNumber-1]->PDFiltering = 1;
    else
        CameraWorkers[CameraNumber-1]->PDFiltering = ui->PDFiltering->isChecked();

    // People detection representation methods
    CameraWorkers[CameraNumber-1]->RepresentationOption = ui->RepresentationCB->currentText().toStdString();

    // Semantic drawing enable
    CameraWorkers[CameraNumber-1]->SemanticDisplay = ui->SemanticEnable->isChecked();

    // Ground truth drawing enable
    CameraWorkers[CameraNumber-1]->GTDisplay = ui->GTDisplay->isChecked();

    // Pedestrian score threshold
    CameraWorkers[CameraNumber-1]->PeopleDetec.Threshold = ui->PDThreshold->value();

    // Pedestrian filtering with multicamera or semantic information
    CameraWorkers[CameraNumber-1]->MultiCameraFiltering = ui->MultiCameraFiltering->isChecked();
    CameraWorkers[CameraNumber-1]->SemanticFiltering = ui->SemanticFiltering->isChecked();
}

void MainWindow::displayFrame(Mat frame, Mat CenitalPlane, int CameraNumber)
{
    if (CameraNumber == 1) {
        // Camera 1
        ui->CVWidget1->showImage(frame);
    }
    else if (CameraNumber == 2) {
        // Camera 2
        ui->CVWidget2->showImage(frame);
    }
    else if (CameraNumber == 3) {
        // Camera 3
        ui->CVWidget3->showImage(frame);
        // Cenital Plane is only displayed once
        ui->CVWidgetCenital->showImage(CenitalPlane);
    }

    // Other Displays
    /*
    Mat SemanticMask1 =  CameraWorkers[0]->SemanticMask;
    Mat SemanticMask2 =  CameraWorkers[1]->SemanticMask;
    Mat SemanticMask3 =  CameraWorkers[2]->SemanticMask;

    imshow("Semantic Camera 1", SemanticMask1);
    imshow("Semantic Camera 2", SemanticMask2);
    imshow("Semantic Camera 3", SemanticMask3);

    Mat SemanticMask1 =  CameraWorkers[0]->Camera.CommonSemantic12;
    Mat SemanticMask2 =  CameraWorkers[0]->Camera.CommonSemantic23;
    Mat SemanticMask3 =  CameraWorkers[0]->Camera.CommonSemantic13;

    imshow("Common Semantic between Camera 1 and 2", SemanticMask1*20);
    imshow("Common Semantic between Camera 2 and 3", SemanticMask2*20);
    imshow("Common Semantic between Camera 1 and 3", SemanticMask3*20);

    Mat CommonSemantic = CameraWorkers[0]->Camera.CommonSemanticAllCameras;
    imshow("Common Semantic between all cameras", CommonSemantic*20);
    */
}

void MainWindow::joinCenitalFrames(Mat frame, Mat CenitalPlane, int CameraNumber)
{
    Mat CenitalFrame1 =  CameraWorkers[0]->CenitalPlane;
    Mat CenitalFrame2 =  CameraWorkers[1]->CenitalPlane;
    Mat CenitalFrame3 =  CameraWorkers[2]->CenitalPlane;

    // Join all the images
    add(CenitalPlane, CenitalFrame1, CenitalPlane);
    add(CenitalPlane, CenitalFrame2, CenitalPlane);
    add(CenitalPlane, CenitalFrame3, CenitalPlane);
    emit cenitalJoined(frame, CenitalPlane, CameraNumber);
}

void MainWindow::sharePedestrianDetections(int CameraNumber)
{
    if (CameraNumber == 1) {
        // Clear vectors
        CameraWorkers[CameraNumber-1]->ProjCenterPoints1.clear();
        CameraWorkers[CameraNumber-1]->ProjCenterPoints2.clear();
        CameraWorkers[CameraNumber-1]->ProjLeftPoints1.clear();
        CameraWorkers[CameraNumber-1]->ProjLeftPoints2.clear();
        CameraWorkers[CameraNumber-1]->ProjRightPoints1.clear();
        CameraWorkers[CameraNumber-1]->ProjRightPoints2.clear();

        // Copy projections from camera 2
        CameraWorkers[CameraNumber-1]->ProjCenterPoints1 = CameraWorkers[1]->PeopleDetec.ProjectedCenterPoints;
        CameraWorkers[CameraNumber-1]->ProjLeftPoints1   = CameraWorkers[1]->PeopleDetec.ProjectedLeftPoints;
        CameraWorkers[CameraNumber-1]->ProjRightPoints1  = CameraWorkers[1]->PeopleDetec.ProjectedRightPoints;

        // Copy projections from camera 3
        CameraWorkers[CameraNumber-1]->ProjCenterPoints2 = CameraWorkers[2]->PeopleDetec.ProjectedCenterPoints;
        CameraWorkers[CameraNumber-1]->ProjLeftPoints2   = CameraWorkers[2]->PeopleDetec.ProjectedLeftPoints;
        CameraWorkers[CameraNumber-1]->ProjRightPoints2  = CameraWorkers[2]->PeopleDetec.ProjectedRightPoints;
    }
    else if (CameraNumber == 2) {
        // Clear vectors
        CameraWorkers[CameraNumber-1]->ProjCenterPoints1.clear();
        CameraWorkers[CameraNumber-1]->ProjCenterPoints2.clear();
        CameraWorkers[CameraNumber-1]->ProjLeftPoints1.clear();
        CameraWorkers[CameraNumber-1]->ProjLeftPoints2.clear();
        CameraWorkers[CameraNumber-1]->ProjRightPoints1.clear();
        CameraWorkers[CameraNumber-1]->ProjRightPoints2.clear();

        // Copy projections from camera 1
        CameraWorkers[CameraNumber-1]->ProjCenterPoints1 = CameraWorkers[0]->PeopleDetec.ProjectedCenterPoints;
        CameraWorkers[CameraNumber-1]->ProjLeftPoints1   = CameraWorkers[0]->PeopleDetec.ProjectedLeftPoints;
        CameraWorkers[CameraNumber-1]->ProjRightPoints1  = CameraWorkers[0]->PeopleDetec.ProjectedRightPoints;

        // Copy projections from camera 3
        CameraWorkers[CameraNumber-1]->ProjCenterPoints2 = CameraWorkers[2]->PeopleDetec.ProjectedCenterPoints;
        CameraWorkers[CameraNumber-1]->ProjLeftPoints2   = CameraWorkers[2]->PeopleDetec.ProjectedLeftPoints;
        CameraWorkers[CameraNumber-1]->ProjRightPoints2  = CameraWorkers[2]->PeopleDetec.ProjectedRightPoints;
    }
    else if (CameraNumber == 3) {
        // Clear vectors
        CameraWorkers[CameraNumber-1]->ProjCenterPoints1.clear();
        CameraWorkers[CameraNumber-1]->ProjCenterPoints2.clear();
        CameraWorkers[CameraNumber-1]->ProjLeftPoints1.clear();
        CameraWorkers[CameraNumber-1]->ProjLeftPoints2.clear();
        CameraWorkers[CameraNumber-1]->ProjRightPoints1.clear();
        CameraWorkers[CameraNumber-1]->ProjRightPoints2.clear();

        // Copy projections from camera 1
        CameraWorkers[CameraNumber-1]->ProjCenterPoints2 = CameraWorkers[0]->PeopleDetec.ProjectedCenterPoints;
        CameraWorkers[CameraNumber-1]->ProjLeftPoints2   = CameraWorkers[0]->PeopleDetec.ProjectedLeftPoints;
        CameraWorkers[CameraNumber-1]->ProjRightPoints2  = CameraWorkers[0]->PeopleDetec.ProjectedRightPoints;

        // Copy projections from camera 2
        CameraWorkers[CameraNumber-1]->ProjCenterPoints1 = CameraWorkers[1]->PeopleDetec.ProjectedCenterPoints;
        CameraWorkers[CameraNumber-1]->ProjLeftPoints1   = CameraWorkers[1]->PeopleDetec.ProjectedLeftPoints;
        CameraWorkers[CameraNumber-1]->ProjRightPoints1  = CameraWorkers[1]->PeopleDetec.ProjectedRightPoints;
    }
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
        PtsDstFile << pt.x << " " << pt.y << endl;
    }
}

void onMouseCamera1Frame(int evt, int x, int y, int, void*)
{
    if(evt == CV_EVENT_LBUTTONDOWN) {
        Point pt = Point(x,y);
        cout << "Camera 1 Frame x = " << pt.x << " y = " << pt.y << endl;
        // Save points into txt
        PtsSrcFile << pt.x << " " << pt.y << endl;
    }
}

void MainWindow::on_actionCamera_1_triggered()
{
    string VideoPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5";
    for(int i = 1; i <= NViews; i++){
        // Load the cenital plane
        Mat CenitalFrame = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/CenitalViewMeasured.png");
        Mat CameraFrame = imread(VideoPath + "/Homography Images/Camera 1/View " + to_string(i) + ".jpg");

        // Open files
        PtsDstFile.open(VideoPath + "/Homography Images/Camera 1/View " + to_string(i) + "_PtsDstFile.txt");
        PtsSrcFile.open(VideoPath + "/Homography Images/Camera 1/View " + to_string(i) + "_PtsSrcFile.txt");

        String CenitalWindow = "Cenital Frame";
        namedWindow(CenitalWindow);
        setMouseCallback(CenitalWindow, onMouseCamera1Cenital, 0);
        imshow(CenitalWindow, CenitalFrame);

        String FrameWindow = "Camera 1 Frame. View " + to_string(i);
        namedWindow(FrameWindow);
        setMouseCallback(FrameWindow, onMouseCamera1Frame, 0);
        imshow(FrameWindow, CameraFrame);

        if(waitKey()==27) {
            PtsDstFile.close();
            PtsSrcFile.close();
            destroyWindow(CenitalWindow);
            destroyWindow(FrameWindow);
        }
    }
    return;
}

void onMouseCamera2Cenital(int evt, int x, int y, int, void*)
{
    if(evt == CV_EVENT_LBUTTONDOWN) {
        Point pt = Point(x,y);
        cout << "Cenital Frame x = " << pt.x << " y = " << pt.y << endl;
        // Save Points into txt file
        PtsDstFile << pt.x << " " << pt.y << endl;
    }
}

void onMouseCamera2Frame(int evt, int x, int y, int, void*)
{
    if(evt == CV_EVENT_LBUTTONDOWN) {
        Point pt = Point(x,y);
        cout << "Camera 2 Frame x = " << pt.x << " y = " << pt.y << endl;
        // Save points into txt
        PtsSrcFile << pt.x << " " << pt.y << endl;
    }
}

void MainWindow::on_actionCamera_2_triggered()
{
    string VideoPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5";
    for(int i = 1; i <= NViews; i++){
        // Load the cenital plane
        Mat CenitalFrame = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/CenitalViewMeasured.png");
        Mat CameraFrame = imread(VideoPath + "/Homography Images/Camera 2/View " + to_string(i) + ".jpg");

        // Open files
        PtsDstFile.open(VideoPath + "/Homography Images/Camera 2/View " + to_string(i) + "_PtsDstFile.txt");
        PtsSrcFile.open(VideoPath + "/Homography Images/Camera 2/View " + to_string(i) + "_PtsSrcFile.txt");

        String CenitalWindow = "Cenital Frame";
        namedWindow(CenitalWindow);
        setMouseCallback(CenitalWindow, onMouseCamera2Cenital, 0);
        imshow(CenitalWindow, CenitalFrame);

        String FrameWindow = "Camera 2 Frame. View " + to_string(i);
        namedWindow(FrameWindow);
        setMouseCallback(FrameWindow, onMouseCamera2Frame, 0);
        imshow(FrameWindow, CameraFrame);

        if(waitKey() == 27) {
            PtsDstFile.close();
            PtsSrcFile.close();
            destroyWindow(CenitalWindow);
            destroyWindow(FrameWindow);
        }
    }
    return;
}

void onMouseCamera3Cenital(int evt, int x, int y, int, void*)
{
    if(evt == CV_EVENT_LBUTTONDOWN) {
        Point pt = Point(x,y);
        cout << "Cenital Frame x = " << pt.x << " y = " << pt.y << endl;
        // Save Points into txt file
        PtsDstFile << pt.x << " " << pt.y << endl;
    }
}

void onMouseCamera3Frame(int evt, int x, int y, int, void*)
{
    if(evt == CV_EVENT_LBUTTONDOWN) {
        Point pt = Point(x,y);
        cout << "Camera 3 Frame x = " << pt.x << " y = " << pt.y << endl;
        // Save points into txt
        PtsSrcFile << pt.x << " " << pt.y << endl;
    }
}

void MainWindow::on_actionCamera_3_triggered()
{
    string VideoPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5";
    for(int i = 1; i <= NViews; i++){
        // Load the cenital plane
        Mat CenitalFrame = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/CenitalViewMeasured.png");
        Mat CameraFrame = imread(VideoPath + "/Homography Images/Camera 3/View " + to_string(i) + ".jpg");

        // Open files
        PtsDstFile.open(VideoPath + "/Homography Images/Camera 3/View " + to_string(i) + "_PtsDstFile.txt");
        PtsSrcFile.open(VideoPath + "/Homography Images/Camera 3/View " + to_string(i) + "_PtsSrcFile.txt");

        String CenitalWindow = "Cenital Frame";
        namedWindow(CenitalWindow);
        setMouseCallback(CenitalWindow, onMouseCamera3Cenital, 0);
        imshow(CenitalWindow, CenitalFrame);

        String FrameWindow = "Camera 3 Frame. View " + to_string(i);
        namedWindow(FrameWindow);
        setMouseCallback(FrameWindow, onMouseCamera3Frame, 0);
        imshow(FrameWindow, CameraFrame);

        if(waitKey() == 27) {
            PtsDstFile.close();
            PtsSrcFile.close();
            destroyWindow(CenitalWindow);
            destroyWindow(FrameWindow);
        }
    }
    return;
}

void MainWindow::on_actionAbout_triggered()
{
    AboutDialog Aboutwindow;
    Aboutwindow.setModal(true);
    Aboutwindow.exec();
}
