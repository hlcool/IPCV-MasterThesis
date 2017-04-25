#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "barrier.h"
#include "camerastream.h"
#include "cameraworker.h"

#include <QDebug>
#include <QThread>
#include <QLabel>
#include <QGridLayout>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <boost/lexical_cast.hpp>
#include <QFileDialog>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

vector<CameraStream> Cameras;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

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
        // Opens the QFileDialog
        filenames = QFileDialog::getOpenFileNames(this, tr("Open video files for the cameras"), GlobalPath, tr("Video Files (*.mpg *.avi *.m4v *.ts *.m2v)"));
    }
    else {
        filenames << "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Videos/Camera1Sync.m2v";
        filenames << "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Videos/Camera2Sync.m2v";
        filenames << "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Videos/Camera3Sync.m2v";
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

        // Fill UI Variables in the CameraWorker
        CameraWorkers[i]->WidgetHeight = ui->CVWidget1->height();
        CameraWorkers[i]->WidgetWidth = ui->CVWidget1->width();
        CameraWorkers[i]->CBOption =  ui->PeopleDetectorCB->currentText().toStdString();
        // Representation Method
        CameraWorkers[i]->RepresentationOption = ui->RepresentationCB->currentText().toStdString();
        CameraWorkers[i]->PDFiltering = ui->PDFiltering->isChecked();

        // FastRCNN Method
        if (ui->FastButton->isChecked())
            CameraWorkers[i]->FastRCNNMethod = "fast";
        else if (ui->AccurateButton->isChecked())
            CameraWorkers[i]->FastRCNNMethod = "accurate";

        // Move CameraWorker to thread
        CameraWorkers[i]->moveToThread(threads[i]);

        // Connect signals to slot between thread and CameraWorkers
        connectSignals2Slots(threads[i], CameraWorkers[i]);

        // Thread is started
        threads[i]->start();
        qDebug() << "Thread from camera " << (i+1) << " started";
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
    // frameFinished signal with updateVariables and joinCenitalFrames
    connect(worker, SIGNAL(frameFinished(Mat, Mat, int)), this, SLOT(updateVariables(Mat, Mat, int)));
    connect(worker, SIGNAL(frameFinished(Mat, Mat, int)), this, SLOT(joinCenitalFrames(Mat, Mat, int)));
    // cenitalJoined signal with displayFrame
    connect(this, SIGNAL(cenitalJoined(Mat, Mat, int)), this, SLOT(displayFrame(Mat, Mat, int)));

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
    if(CameraNumber == 1){
        CameraWorkers[0]->WidgetHeight = ui->CVWidget1->height();
        CameraWorkers[0]->WidgetWidth = ui->CVWidget1->width();
    }
    if(CameraNumber == 2){
        CameraWorkers[1]->WidgetHeight = ui->CVWidget1->height();
        CameraWorkers[1]->WidgetWidth = ui->CVWidget1->width();
    }
    if(CameraNumber == 3){
        CameraWorkers[2]->WidgetHeight = ui->CVWidget1->height();
        CameraWorkers[2]->WidgetWidth = ui->CVWidget1->width();
    }
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
        // CenitalPlane.convertTo(CenitalPlane, CV_8UC3, 255.0);
        // Cenital Plane is only displayed once
        ui->CVWidgetCenital->showImage(CenitalPlane);
    }
}

void MainWindow::joinCenitalFrames(Mat frame, Mat CenitalPlane, int CameraNumber)
{
   Mat CenitalFrame1, CenitalFrame2, CenitalFrame3;

   CenitalFrame1 =  CameraWorkers[0]->CenitalPlane;
   CenitalFrame2 =  CameraWorkers[1]->CenitalPlane;
   CenitalFrame3 =  CameraWorkers[2]->CenitalPlane;

   // Join all the images
   add(CenitalPlane, CenitalFrame1, CenitalPlane);
   add(CenitalPlane, CenitalFrame2, CenitalPlane);
   add(CenitalPlane, CenitalFrame3, CenitalPlane);
   emit cenitalJoined(frame, CenitalPlane, CameraNumber);
}
