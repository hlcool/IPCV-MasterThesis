#include "cameraworker.h"
#include "camerastream.h"
#include "barrier.h"
#include "peopledetector.h"
#include <QDebug>
#include <QThread>
#include <QTime>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

CameraWorker::CameraWorker(CameraStream Camera, Barrier barrier) : Camera(Camera), barrier(barrier) {}

CameraWorker::~CameraWorker(){}

void CameraWorker::preProcessVideo()
{
    // Open video file
    Camera.VideoOpenning(Camera.InputPath);

    // Display information
    cout << "Camera " << Camera.CameraNumber << " opened correctly"  << endl;
    cout << "The video to process has the following information:" << endl;
    cout << "Width: " << Camera.Width << ". Heigth: " << Camera.Height << ". Frames/second: " << Camera.FrameRate << endl;
    cout << "The total number of frames is: " << Camera.FrameNumber << endl;
    cout << "" << endl;

    // Create and open the statistics file
    VideoStatsFile.open("/Users/alex/IPCV-MasterThesis/MultithreadApplication/VideoProcessingStats" + to_string(Camera.CameraNumber) + ".txt");
    VideoStatsFile << "Frame  Computational Time" << endl;

    // Compute camera homographies
    Camera.computeHomography();

    // Compute AKAZE points for camera views
    Camera.AkazePointsForViewImages();

    // Main video processing function
    processVideo();
}

void CameraWorker::processVideo()
{
    // People detector class
    PeopleDetector PeopleDetec;

    // Main Video Loop
    while (true) {
        // Start the clock for measuring time consumption/frame
        clock_t begin = clock();

        // Extract ActualFrame
        Camera.cap >> Camera.ActualFrame;
        // Load CenitalImage
        Mat CenitalPlaneImage = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/CenitalViewMeasured.png");

        if (Camera.ActualFrame.empty()) {
            // Empty frame to display when the video has finished
            Camera.ActualFrame = Mat(Size(720, 576), CV_8UC3, Scalar(192, 0, 0));
            // Emit signal frameFinished
            emit frameFinished(Camera.ActualFrame, CenitalPlaneImage, Camera.CameraNumber);
            // Close time consumption file
            VideoStatsFile.close();
            break;
        }

        // Get frame number
        stringstream ss;
        ss << Camera.cap.get(CAP_PROP_POS_FRAMES);
        String FrameNumber = ss.str().c_str();

        /* -----------------------*/
        /*      MAIN ALGORITHM    */
        /* -----------------------*/

        // -------------------------------- //
        //   MASK EXTRACTION AND FILTERING  //
        // -------------------------------- //
        // Load actual semantic frame
        Camera.getActualSemFrame(FrameNumber);

        if (PDFiltering) {
            // Extract pedestrian mask with semantic information
            Camera.extractPDMask(Camera.ActualSemFrame);
            // Extract blobs from the previous mask
            Camera.extractFGBlobs(Camera.PedestrianMask, CBOption);
            // Extract images from blobs
            Camera.ExtractFGImages(Camera.ActualFrame, Camera.FGBlobs);
        }

        // ----------------------- //
        //   HOMOGRAPHY SELECTION  //
        // ----------------------- //
        Camera.HomographySelection(Camera.HomographyVector);

        // ----------------------- //
        //   SEMANTIC PROJECTION   //
        // ----------------------- //
        // Auxiliar Cenital plane to paint
        CenitalPlane = Mat::zeros(CenitalPlaneImage.rows, CenitalPlaneImage.cols, CenitalPlaneImage.type());
        // Project Floor Points
        Camera.ProjectFloorPoints(CenitalPlane);
        // Draw semantic projection
        Camera.drawSemantic(CenitalPlane);

        // ------------------------------------------- //
        //     PEOPLE DETECTION & BLOBS PROJECTION     //
        // ------------------------------------------- //
        PeopleDetec.MainPeopleDetection(Camera, CBOption, RepresentationOption, PDFiltering, CenitalPlane);

        // ------------------------------------------- //
        //        FRAME RESIZE AND FRAME NUMBER        //
        // ------------------------------------------- //
        // Resize the frames accordingly to the widgets size
        cv::resize(Camera.ActualFrame, Camera.ActualFrame, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);
        cv::resize(CenitalPlane, CenitalPlane, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);
        cv::resize(CenitalPlaneImage, CenitalPlaneImage, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);

        // Write frame number on the camera frame
        putText(Camera.ActualFrame, FrameNumber, Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

        // Compute the processing time per frame
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

        // Save measures to .txt file
        VideoStatsFile << FrameNumber << "       " << elapsed_secs << endl;

        //qDebug() << "Thread " << Camera.CameraNumber << " processing frame " << QString::fromStdString(FrameNumber);
        // Threads must wait here until all of them have reached the barrier
        barrier.wait();
        emit frameFinished(Camera.ActualFrame.clone(), CenitalPlaneImage.clone(), Camera.CameraNumber);
    }
    emit finished();
}
