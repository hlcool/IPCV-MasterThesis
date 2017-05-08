#include "cameraworker.h"
#include "camerastream.h"
#include "barrier.h"
#include "peopledetector.h"
#include <QDebug>
#include <string>
#include <QThread>
#include <iostream>
#include <QElapsedTimer>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

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

    // Extract common projected semantic points
    Camera.SemanticCommonPoints();

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
        QElapsedTimer timer;
        timer.start();

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

        // ------------------------------ //
        //   HOMOGRAPHY & VIEW SELECTION  //
        // ------------------------------ //
        Camera.ViewSelection(Camera.HomographyVector);

        // ----------------------- //
        //   SEMANTIC PROJECTION   //
        // ----------------------- //
        // Auxiliar Cenital plane to paint
        CenitalPlane = Mat::zeros(CenitalPlaneImage.rows, CenitalPlaneImage.cols, CenitalPlaneImage.type());
        // Project Floor Points
        SemanticMask = Mat::zeros(CenitalPlaneImage.rows, CenitalPlaneImage.cols, CenitalPlaneImage.type());
        Camera.ProjectSemanticPoints(CenitalPlane, SemanticMask, FrameNumber);
        // Draw semantic projection
        Camera.drawSemantic(CenitalPlane);

        // ---------------------------- //
        //   INDUCED PLANE HOMOGRAPHY   //
        // ---------------------------- //
        Camera.ProjectCommonSemantic();

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
        double mseconds;
        mseconds = timer.elapsed();
        mseconds = mseconds / 1000;

        // Save measures to .txt file
        VideoStatsFile << FrameNumber << "       " << mseconds << endl;

        //qDebug() << "Thread " << Camera.CameraNumber << " processing frame " << QString::fromStdString(FrameNumber);
        // Threads must wait here until all of them have reached the barrier
        barrier.wait();
        emit frameFinished(Camera.ActualFrame, CenitalPlaneImage, Camera.CameraNumber);
    }
    emit finished();
}
