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

CameraWorker::CameraWorker(CameraStream Camera, Barrier barrier) : Camera(Camera), barrier(barrier) {
}

CameraWorker::~CameraWorker(){
}

void CameraWorker::preProcessVideo()
{
    // Open video file
    Camera.VideoOpenning(Camera.InputPath);

    // Create and open the statistics file
    VideoStatsFile.open("/Users/alex/IPCV-MasterThesis/MultithreadApplication/VideoProcessingStats" + to_string(Camera.CameraNumber) + ".txt");
    VideoStatsFile << "Frame  Computational Time" << endl;

    // Compute camera homographies
    Camera.computeHomography();

    // Compute AKAZE points for camera views
    Camera.AkazePointsForViewImages();

    processVideo();
}

void CameraWorker::processVideo()
{
    PeopleDetector PeopleDetec;

    // Main Video Loop
    while (true) {
        // Start the clock for measuring frame consumption
        clock_t begin = clock();

        // Extract ActualFrame
        Camera.cap >> Camera.ActualFrame;
        // Load CenitalImage
        Mat CenitalPlaneImage = imread("/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/CenitalViewMeasured.png");

        // Get frame number
        stringstream ss;
        ss << Camera.cap.get(CAP_PROP_POS_FRAMES);
        String FrameNumber = ss.str().c_str();

        if (Camera.ActualFrame.empty()) {
            // Empty frame to display when the video has finished
            Camera.ActualFrame = Mat(Size(720, 576), CV_8UC3, Scalar(192, 0, 0));
            emit frameFinished(Camera.ActualFrame, CenitalPlaneImage, Camera.CameraNumber);
            VideoStatsFile.close();
            qDebug() << "Video finished";
            break;
        }

        Camera.getActualSemFrame(FrameNumber);

        if (PDFiltering) {
            // Compute People detection mask with semantic actual frame
            Camera.extractPDMask(Camera.ActualSemFrame);
            // Extract blobs from the mask
            Camera.extractFGBlobs(Camera.PedestrianMask, CBOption);
            // Extract images from blobs
            Camera.ExtractFGImages(Camera.ActualFrame, Camera.FGBlobs);
        }

        /* -----------------------*/
        /*      MAIN ALGORITHM    */
        /* -----------------------*/

        // ----------------------- //
        //   HOMOGRAPHY SELECTION  //
        // ----------------------- //
        Camera.HomographySelection(Camera.HomographyVector);

        // ----------------------- //
        //   SEMANTIC PROJECTION   //
        // ----------------------- //
        // Auxiliar Cenital plane
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
        // Resize the frame to widget size
        cv::resize(Camera.ActualFrame, Camera.ActualFrame, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);
        cv::resize(CenitalPlane, CenitalPlane, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);
        cv::resize(CenitalPlaneImage, CenitalPlaneImage, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);

        // Extract Frame number and write it on the frame
        putText(Camera.ActualFrame, FrameNumber, Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

        // Compute the processing time per frame
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

        // Save measures to .txt file
        VideoStatsFile << FrameNumber << "       " << elapsed_secs << endl;

        //qDebug() << "Thread " << Camera.CameraNumber << " processing frame " << QString::fromStdString(FrameNumber);

        barrier.wait();
        emit frameFinished(Camera.ActualFrame.clone(), CenitalPlaneImage.clone(), Camera.CameraNumber);
    }
    emit finished();
}



