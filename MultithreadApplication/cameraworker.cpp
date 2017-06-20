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

    // Create and open the statistics file
    Evaluate.EvaluationFile.open("/Users/alex/IPCV-MasterThesis/MultithreadApplication/Evaluation Stats Camera " + to_string(Camera.CameraNumber) + ".txt");
    Evaluate.EvaluationFile << "Frame   GroundTruth Elements   TruePositives  False Positives  Number of Detections   False Negatives    Precision           Recall" << endl;

    // Compute camera homographies
    Camera.computeHomography();

    // Compute AKAZE points for camera views
    Camera.AkazePointsForViewImages();

    // Extract common projected semantic points
    Camera.SemanticCommonPoints();

    // Extract plane scores for views
    Camera.ExtractViewScores();

    // Main video processing function
    processVideo();
}

void CameraWorker::processVideo()
{
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
            Evaluate.EvaluationFile.close();
            exit(EXIT_FAILURE);
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
        if(MultiCameraFiltering || SemanticFiltering){
            Camera.ViewSelection(Camera.HomographyVector);
            Mat ImageWarping = Mat::zeros(CenitalPlaneImage.rows, CenitalPlaneImage.cols, CenitalPlaneImage.type());
            Camera.saveWarpImages(Camera.ActualFrame, Camera.Homography, FrameNumber, ImageWarping);
        }

        // ----------------------- //
        //   SEMANTIC PROJECTION   //
        // ----------------------- //
        // Clear Cenital Plane
        CenitalPlane = Mat::zeros(CenitalPlaneImage.rows, CenitalPlaneImage.cols, CenitalPlaneImage.type());
        // Clear Semantic Mask
        SemanticMask = Mat::zeros(CenitalPlaneImage.rows, CenitalPlaneImage.cols, CenitalPlaneImage.type());
        if(MultiCameraFiltering || SemanticFiltering){
            // Project Semantic to the cenital plane
            Camera.ProjectSemanticPoints(CenitalPlane, SemanticMask, FrameNumber);
            // Draw semantic projection in the cenital plane
            Camera.drawSemantic(CenitalPlane);
        }

        // ------------------------------------------- //
        //     PEOPLE DETECTION & BLOBS PROJECTION     //
        // ------------------------------------------- //
        PeopleDetec.MainPeopleDetection(Camera, FrameNumber, CBOption, RepresentationOption, PDFiltering, CenitalPlane, MultiCameraFiltering, SemanticFiltering);
        if(MultiCameraFiltering){
            barrier.wait();
            emit PedestrianDetectionFinished(Camera.CameraNumber);
        }

        // Reproject to the frames other camera detections
        if(MultiCameraFiltering){
            PeopleDetec.ReprojectionFusion(ProjCenterPoints1, ProjLeftPoints1, ProjRightPoints1, Camera.Homography, Camera.HomographyBetweenViews, Camera.ActualFrame);
            PeopleDetec.ReprojectionFusion(ProjCenterPoints2, ProjLeftPoints2, ProjRightPoints2, Camera.Homography, Camera.HomographyBetweenViews, Camera.ActualFrame);
        }

        // Non Maximum supression betwen blobs from all the cameras
        Camera.non_max_suppresion(PeopleDetec.AllPedestrianVector, PeopleDetec.AllPedestrianVectorNMS);

        // Filter Pedestrian Detections that are not correcly placed within the semantic (floor)
        if(SemanticFiltering)
            PeopleDetec.SemanticConstraining(PeopleDetec.AllPedestrianVectorNMS, Camera.CameraNumber, Camera.ActualFrame, Camera.Homography, Camera.HomographyBetweenViews);

        // Paint Bounding Boxes
        PeopleDetec.paintBoundingBoxes(Camera.ActualFrame, CBOption, PeopleDetec.AllPedestrianVectorNMS, Camera.CameraNumber, 2);

        // ---------------------------------- //
        //             DATA USAGE             //
        // ---------------------------------- //
        if(MultiCameraFiltering || SemanticFiltering){
                PeopleDetec.ExtractDataUsage(Camera.CameraNumber, FrameNumber, Camera.Homography, Camera.HomographyBetweenViews);
        }

        // ---------------------------------- //
        //             EVALUATION             //
        // ---------------------------------- //
        vector<Rect> GroundTruthVector;
        Evaluate.GTTextParser(Camera.CameraNumber, GroundTruthVector, FrameNumber);
        if(GTDisplay)
            PeopleDetec.paintBoundingBoxes(Camera.ActualFrame, "GT", GroundTruthVector, Camera.CameraNumber, 5);
        Evaluate.ExtractEvaluationScores(GroundTruthVector, PeopleDetec.AllPedestrianVectorNMS, FrameNumber);

        // ---------------------------- //
        //   INDUCED PLANE HOMOGRAPHY   //
        // ---------------------------- //
        // Project common semantic back to the camera frames
        if(SemanticDisplay)
            Camera.ProjectCommonSemantic();

        // ------------------------------------------- //
        //        FRAME RESIZE AND FRAME NUMBER        //
        // ------------------------------------------- //
        // Resize the frames accordingly to the widgets size
        cv::resize(Camera.ActualFrame, Camera.ActualFrame, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);
        if(MultiCameraFiltering || SemanticFiltering){
            cv::resize(CenitalPlane, CenitalPlane, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);
            cv::resize(CenitalPlaneImage, CenitalPlaneImage, {WidgetWidth, WidgetHeight}, INTER_LANCZOS4);
        }

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
