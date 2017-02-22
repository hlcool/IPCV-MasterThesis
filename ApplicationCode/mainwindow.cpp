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


using namespace std;
using namespace cv;

VideoFile Video;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    cout << "Open a video file" << endl;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionOpen_file_triggered()
{

    // Global Path variable should be change if used in other computer
    QString GlobalPath = "/Users/alex/IPCV-MasterThesis/ApplicationCode";

    if (ProgramFlag) {
        // Get a filename to open
        QString filePath = QFileDialog::getOpenFileName(this, tr("Open Image"), GlobalPath, tr("Video Files (*.mpg *.avi *.m4v *.ts)"));
        // Convert QString to std::string
        Video.InputPath = filePath.toStdString();
    }
    else {
        Video.InputPath = GlobalPath.toStdString() + "/Inputs/HallCutted.mpg";
    }

    Video.VideoOpenning(Video.InputPath);

    cout << "Video opened correctly"  << endl;
    cout << "Video processing starts" << endl;

    // Create and open the statistics file
    Video.VideoStatsFile.open(GlobalPath.toStdString() + "/VideoProcessingStats.txt");
    Video.VideoStatsFile << "Frame  Computational Time" << endl;

    // Timer to launch the ProcessVideo() slot
    imageTimer = new QTimer(this);
    connect(imageTimer, SIGNAL(timeout()), this, SLOT(ProcessVideo()));
    imageTimer->start();

}


void MainWindow::ProcessVideo(){

    // Start the clock for measuring frame consumption
    clock_t begin = clock();

    Video.cap >> Video.ActualFrame; // Get first video frame

    stringstream ss;
    ss << Video.cap.get(CAP_PROP_POS_FRAMES);

    // Check if we achieved the end of the file (e.g. ActualFrame.data is empty)
    if (!Video.ActualFrame.data){
        cout << "The processing has finished" << endl;
        Video.VideoStatsFile.close();
        imageTimer->blockSignals(true);
        return;
    }


    /* -----------------------*/
    /*      MAIN ALGORITHM    */
    /* -----------------------*/

    // ----------------------- //
    //       ENHANCEMENT       //
    // ----------------------- //

    // Frames Enhancement
    Video.imageEnhancement(Video.ActualFrame);

    // ----------------------- //
    // BACKGROUND SUBSTRACTION //
    // ----------------------- //

    // Compute Background Mask
    Video.pMOG2->apply(Video.ActualFrame, Video.BackgroundMask);
    // Improve Background Mask
    Video.maskEnhancement(Video.BackgroundMask);

    // ----------------------- //
    //     HOMOGRAPHY & IW     //
    // ----------------------- //

    Video.computeHomography();
    //Video.ImageWarping = Mat::zeros(480, 1280, CV_64F);
    //warpPerspective(Video.ActualFrame2, Video.ImageWarping, Video.Homography, Video.ImageWarping.size());
    //imshow("Warped Image", Video.ImageWarping);

    // ----------------------- //
    //     PEOPLE DETECTION    //
    // ----------------------- //

    String CBOption = ui->PeopleDetectorCB->currentText().toStdString();
    if (ui->FastButton->isChecked())
        Video.FastRCNNMethod = "fast";
    else if (ui->AccurateButton->isChecked())
        Video.FastRCNNMethod = "accurate";

    if (!CBOption.compare("Histogram of Oriented Gradients")){
        // HOG Detector
        Video.HOGPeopleDetection(Video.ActualFrame);
        Video.projectBlobs(Video.HOGBoundingBoxesNMS, Video.HOGScores, Video.Homography, "GREEN");
    }
    else if(!CBOption.compare("FastRCNN")){
        // FastRCNN Detector
        Video.FastRCNNPeopleDetection(ss.str(), Video.FastRCNNMethod);
        Video.projectBlobs(Video.RCNNBoundingBoxesNMS, Video.RCNNScores, Video.Homography, "RED");
    }
    else{
        // HOG Detector
        Video.HOGPeopleDetection(Video.ActualFrame);
        Video.projectBlobs(Video.HOGBoundingBoxesNMS, Video.HOGScores, Video.Homography, "GREEN");

        // FastRCNN Detector
        Video.FastRCNNPeopleDetection(ss.str(), Video.FastRCNNMethod);
        Video.projectBlobs(Video.RCNNBoundingBoxesNMS, Video.RCNNScores, Video.Homography, "RED");
    }

    // ----------------------- //
    //         DISPLAY         //
    // ----------------------- //

    // Paint blobs
    Video.paintBoundingBoxes(Video.ActualFrame, CBOption);

    // Display projected points
    imshow("Projected points", Video.CenitalPlane);

    /* -----------------------*/
    /* -----------------------*/


    // Resize the video for displaying to the size of the widget
    WidgetHeight = ui->CVWidget->height();
    WidgetWidth  = ui->CVWidget->width();
    cv::resize(Video.ActualFrame, Video.ActualFrame, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);
    cv::resize(Video.BackgroundMask, Video.BackgroundMask, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);

    // Extract Frame number and write it on the frame
    putText(Video.ActualFrame, ss.str().c_str(), Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

    // Method to display the frame in the CVWidget
    ui->CVWidget->showImage(Video.ActualFrame);
    ui->CVWidget2->showImage(Video.BackgroundMask);

    // Pause to control the frame rate of the video when the option button is checked
    if (ui->RealTimeButton->isChecked())
        waitKey(30);

    // Compute the processing time per frame
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    // Save measures to .txt file
    Video.VideoStatsFile << ss.str() << "       " << elapsed_secs << endl;

}
