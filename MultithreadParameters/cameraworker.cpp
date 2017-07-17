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

CameraWorker::CameraWorker(CameraStream Camera, Barrier barrier, String ParamPDDetector, double ParamPDThreshold, bool ParamSemanticFitlering, bool ParamMultiCamera) :
    Camera(Camera), barrier(barrier){

    CBOption = ParamPDDetector;
    MultiCameraFiltering = ParamMultiCamera;
    SemanticFiltering = ParamSemanticFitlering;
    PeopleDetec.Threshold = ParamPDThreshold;
}

CameraWorker::~CameraWorker(){}

void CameraWorker::preProcessVideo()
{
    String EvalPath, DetectionsPath;

    if(Camera.CameraNumber == 1){
        cout << endl;
        cout << endl;
        cout << "VIDEO PROCESSING STARTS: " << endl;
        cout << "INFORMATION: " << endl;
        if(SemanticFiltering & MultiCameraFiltering){
            cout << endl;
            cout << "Processing starting with " << CBOption << " People Detector with Th = " << to_string(PeopleDetec.Threshold) << endl;
            cout << "Using Multicamera and semantic" << endl;
            cout << endl;
        }
        else if(MultiCameraFiltering){
            cout << endl;
            cout << "Processing starting with " << CBOption << " People Detector with Th = " << to_string(PeopleDetec.Threshold) << endl;
            cout << "Using Multicamera" << endl;
            cout << endl;
        }
        else if(SemanticFiltering){
            cout << endl;
            cout << "Processing starting with " << CBOption << " People Detector with Th = " << to_string(PeopleDetec.Threshold) << endl;
            cout << "Using semantic" << endl;
            cout << endl;
        }
        else{
            cout << endl;
            cout << "Processing starting with " << CBOption << " People Detector with Th = " << to_string(PeopleDetec.Threshold) << endl;
            cout << "Raw Detector" << endl;
            cout << endl;
        }
    }

    if(SemanticFiltering & MultiCameraFiltering){
        EvalPath = "/Users/alex/IPCV-MasterThesis/MultithreadParameters/Ficheros/4MultiSemantic/" + CBOption + "/Evaluation Stats Camera " + to_string(Camera.CameraNumber) + "-" + to_string(PeopleDetec.Threshold)  + ".txt";
        DetectionsPath = "/Users/alex/IPCV-MasterThesis/Matlab/Evaluation/Video/Pedestrian Detections/4MultiSemantic/BoundingBoxes "+ CBOption + " " + to_string(Camera.CameraNumber) + ".idl";
    }
    else if(MultiCameraFiltering){
        EvalPath = "/Users/alex/IPCV-MasterThesis/MultithreadParameters/Ficheros/2Multi/" + CBOption + "/Evaluation Stats Camera " + to_string(Camera.CameraNumber) + "-" + to_string(PeopleDetec.Threshold)  + ".txt";
        DetectionsPath = "/Users/alex/IPCV-MasterThesis/Matlab/Evaluation/Video/Pedestrian Detections/2Multi/BoundingBoxes "+ CBOption + " " + to_string(Camera.CameraNumber) + ".idl";
    }
    else if(SemanticFiltering){
        EvalPath = "/Users/alex/IPCV-MasterThesis/MultithreadParameters/Ficheros/3Semantic/" + CBOption + "/Evaluation Stats Camera " + to_string(Camera.CameraNumber) + "-" + to_string(PeopleDetec.Threshold)  + ".txt";
        DetectionsPath = "/Users/alex/IPCV-MasterThesis/Matlab/Evaluation/Video/Pedestrian Detections/3Semantic/BoundingBoxes "+ CBOption + " " + to_string(Camera.CameraNumber) + ".idl";
    }
    else{
        EvalPath = "/Users/alex/IPCV-MasterThesis/MultithreadParameters/Ficheros/1Raw/" + CBOption + "/Evaluation Stats Camera " + to_string(Camera.CameraNumber) + "-" + to_string(PeopleDetec.Threshold) + ".txt";
        DetectionsPath = "/Users/alex/IPCV-MasterThesis/Matlab/Evaluation/Video/Pedestrian Detections/1Raw/BoundingBoxes "+ CBOption + " " + to_string(Camera.CameraNumber) + ".idl";
    }

    // Open video file
    Camera.VideoOpenning(Camera.InputPath);

    // Create and open the statistics file
    VideoStatsFile.open("/Users/alex/IPCV-MasterThesis/MultithreadParameters/VideoProcessingStats" + to_string(Camera.CameraNumber) + ".txt");
    VideoStatsFile << "Frame  Computational Time" << endl;

    // Create and open the statistics file
    Evaluate.EvaluationFile.open(EvalPath);
    Evaluate.EvaluationFile << "Frame   GroundTruth Elements   TruePositives  False Positives  Number of Detections   False Negatives    Precision           Recall" << endl;

    // Create and open bounding boxes file
    PeopleDetec.BoundingBoxesFile.open(DetectionsPath);

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
            // Close txt files
            VideoStatsFile.close();
            SelectedViewsFile.close();
            Evaluate.EvaluationFile.close();
            PeopleDetec.BoundingBoxesFile.close();
            exit(EXIT_FAILURE);
            break;
        }

        // Get frame number
        stringstream ss;
        ss << Camera.cap.get(CAP_PROP_POS_FRAMES);
        String FrameNumber = ss.str().c_str();

        if(Camera.CameraNumber == 1){
            if(((stoi(FrameNumber)) % 50) == 0){
                cout << "\r" << "Processing Frame: " << FrameNumber << flush;
            }
        }

        /* -----------------------*/
        /*      MAIN ALGORITHM    */
        /* -----------------------*/
        // Load actual semantic frame
        Camera.getActualSemFrame(FrameNumber);

        // -------------------------------- //
        //   MASK EXTRACTION AND FILTERING  //
        // -------------------------------- //
        // Extract pedestrian mask with semantic information
        Camera.extractPDMask(Camera.ActualSemFrame);
        // Extract blobs from the previous mask
        Camera.extractFGBlobs(Camera.PedestrianMask, CBOption);

        // ------------------------------ //
        //   HOMOGRAPHY & VIEW SELECTION  //
        // ------------------------------ //
        if(MultiCameraFiltering || SemanticFiltering){
            Camera.ViewSelectionFromTXT(Camera.HomographyVector, FrameNumber);
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
        }

        // ------------------------------------------- //
        //     PEOPLE DETECTION & BLOBS PROJECTION     //
        // ------------------------------------------- //
        PeopleDetec.MainPeopleDetection(Camera, FrameNumber, CBOption, RepresentationOption, CenitalPlane, MultiCameraFiltering, SemanticFiltering);
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
        Camera.non_max_suppresion_scores(PeopleDetec.AllPedestrianVector, PeopleDetec.AllPedestrianVectorScore, PeopleDetec.AllPedestrianVectorNMS, PeopleDetec.AllPedestrianVectorScoreNMS);

        // Filter Pedestrian Detections that are not correcly placed within the semantic (floor)
        if(SemanticFiltering)
            PeopleDetec.SemanticConstraining(PeopleDetec.AllPedestrianVectorNMS, PeopleDetec.AllPedestrianVectorScoreNMS, Camera.CameraNumber, Camera.ActualFrame, Camera.Homography, Camera.HomographyBetweenViews);

        // -------------------- //
        //     BLOBS SAVING     //
        // -------------------- //
        PeopleDetec.blobSavingTXT(PeopleDetec.AllPedestrianVectorNMS, PeopleDetec.AllPedestrianVectorScoreNMS, FrameNumber, Camera.CameraNumber);

        // ---------------------------------- //
        //             EVALUATION             //
        // ---------------------------------- //
        //vector<Rect> GroundTruthVector;
        //Evaluate.GTTextParser(Camera.CameraNumber, GroundTruthVector, FrameNumber);
        //Evaluate.ExtractEvaluationScores(GroundTruthVector, PeopleDetec.AllPedestrianVectorNMS, FrameNumber);

        // Compute the processing time per frame
        double mseconds;
        mseconds = timer.elapsed();
        mseconds = mseconds / 1000;

        // Save measures to .txt file
        VideoStatsFile << FrameNumber << "       " << mseconds << endl;

        // Threads must wait here until all of them have reached the barrier
        barrier.wait();
    }
    emit finished();
}
