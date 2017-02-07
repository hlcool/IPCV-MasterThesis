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
        QString filePath = QFileDialog::getOpenFileName(this, tr("Open Image"), GlobalPath, tr("Video Files (*.mpg *.avi)"));
        // Convert QString to std::string
        Video.InputPath = filePath.toStdString();
    }
    else {
        Video.InputPath = GlobalPath.toStdString() + "/Inputs/Vid1.mpg";
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

    Mat ActualFrame;
    Video.cap >> ActualFrame; // Get first video frame

    // Check if we achieved the end of the file (e.g. ActualFrame.data is empty)
    if (!ActualFrame.data){
        cout << "The processing has finished" << endl;
        Video.VideoStatsFile.close();
        imageTimer->blockSignals(true);
        return;
    }


        /* MAIN ALGORITHM */
    /* -----------------------*/



    /* -----------------------*/

    WidgetHeight = ui->CVWidget->height();
    WidgetWidth  = ui->CVWidget->width();

    // Resize the video for displaying to the size of the widget
    cv::resize(ActualFrame, ActualFrame, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);

    // Extract Frame number and write it in red
    stringstream ss;
    ss << Video.cap.get(CAP_PROP_POS_FRAMES);
    putText(ActualFrame, ss.str().c_str(), Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

    ui->CVWidget->showImage(ActualFrame);

    // Pause to control the frame rate of the video when the option button is checked
    if (ui->RealTimeButton->isChecked())
        waitKey(30);

    // Compute the processing time per frame
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    // Save measures to .txt file
    Video.VideoStatsFile << ss.str() << "       " << elapsed_secs << endl;

}
