#include "peopledetector.h"
#include "camerastream.h"
#include <string>
#include <numeric>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

PeopleDetector::PeopleDetector(){}
PeopleDetector::~PeopleDetector(){}

void PeopleDetector::MainPeopleDetection(CameraStream &Camera, String FrameNumber, String CBOption, String RepresentationOption,
                                         bool PDFiltering, Mat &CenitalPlane, bool MultiCameraFiltering, bool SemanticFiltering)
{
    AllPedestrianVector.clear();
    AllPedestrianVectorScore.clear();

    if (!CBOption.compare("HOG")){
        // HOG Detector
        HOGPeopleDetection(Camera);
    }
    else if(!CBOption.compare("FastRCNN")){
        // FastRCNN Detector
        //FastRCNNPeopleDetection(FrameNumber, Camera1.FastRCNNMethod);
    }
    else if(!CBOption.compare("DPM")){
        // DPM Detector
        DPMPeopleDetection(Camera, PDFiltering);
    }
    else if(!CBOption.compare("ACF")){
        // ACF Detector
        ACFPeopleDetection(Camera, PDFiltering);
    }
    else if(!CBOption.compare("Semantic Detector")){
        // People detection using labels from semantic information.
        AllPedestrianVector = Camera.FGBlobs;
        PSPNetScores(Camera.CameraNumber, FrameNumber);
    }
    else if(!CBOption.compare("None")){
        return;
    }

    ThresholdDetections(AllPedestrianVector,  AllPedestrianVectorScore, Threshold);
    if(MultiCameraFiltering || SemanticFiltering)
        projectBlobs(AllPedestrianVector, AllPedestrianVectorScore, Camera.Homography, Camera.HomographyBetweenViews, CenitalPlane, Camera.CameraNumber, RepresentationOption);
}

void PeopleDetector::HOGPeopleDetection(CameraStream &Camera)
{
    // Clear vectors
    Camera.HOGBoundingBoxes.clear();
    Camera.HOGBoundingBoxesNMS.clear();
    Camera.HOGScores.clear();

    // Initialice the SVM
    HOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    // HOG Detector
    vector<double> NoNormalizedScores;
    HOG.detectMultiScale(Camera.ActualFrame, Camera.HOGBoundingBoxes, NoNormalizedScores, 0, Size(8, 8), Size(32, 32), 1.1, 2);

    // Score Normalization
    double MaxHOGScore = 3.8326;
    for(int j = 0; j < NoNormalizedScores.size(); j++){
        double Score = NoNormalizedScores[j];
        double NormalizedScore = Score/MaxHOGScore;
        Camera.HOGScores.push_back(NormalizedScore);
    }

    Camera.HOGBoundingBoxesNMS = Camera.HOGBoundingBoxes;
    AllPedestrianVector = Camera.HOGBoundingBoxesNMS;
    AllPedestrianVectorScore = Camera.HOGScores;
}

void PeopleDetector::DPMPeopleDetection(CameraStream &Camera, bool PDFiltering)
{
    Camera.DPMBoundingBoxes.clear();
    Camera.DPMScores.clear();
    vector<double> NoNormalizedDPMScores;

    // PEOPLE DETECTION FILTERING
    if (PDFiltering){
        if (Camera.FGImages.size() > 0){
            //cout << "Using BKG information" << endl;
            for (size_t i = 0; i < Camera.FGImages.size(); i++) {
                // Auxiliar ActualFrame
                Mat AuxiliarFrame = Camera.FGImages[i].clone();
                //Mat AuxiliarFrame2 = Camara.FGImages[i].clone();
                Rect Offset = Camera.FGBlobs[i];

                if (AuxiliarFrame.rows > 125 && AuxiliarFrame.cols > 125){
                    // Local detection vector
                    vector<DPMDetector::ObjectDetection> DPMBoundingBoxesAux;
                    // DPM detector with NMS. The function destroys the Frame
                    DPMdetector->detect(AuxiliarFrame, DPMBoundingBoxesAux);

                    // Convert from vector<ObjectDetection> to vector<Rect>
                    for (unsigned int i = 0; i < DPMBoundingBoxesAux.size(); i++){
                        Rect Aux1 = DPMBoundingBoxesAux[i].rect;
                        float score = DPMBoundingBoxesAux[i].score;

                        // Convert top-left corner co-ordinates from small image to
                        // complete frame reference
                        Aux1.x = Aux1.x + Offset.x;
                        Aux1.y = Aux1.y + Offset.y;

                        NoNormalizedDPMScores.push_back(score);
                        Camera.DPMBoundingBoxes.push_back(Aux1);

                    }
                }
            }
        }
        else{
            // If there is no person on the semantic mask DPM does not search.
        }
    }
    // PEOPLE DETECTION WITHOUT FILTERING
    else{
        // Auxiliar ActualFrame
        Mat AuxiliarFrame = Camera.ActualFrame.clone();

        // Local detection vector
        vector<DPMDetector::ObjectDetection> DPMBoundingBoxesAux;
        // DPM detector with NMS
        DPMdetector->detect(AuxiliarFrame, DPMBoundingBoxesAux);

        // Convert from vector<ObjectDetection> to vector<Rect>
        for (unsigned int i = 0; i < DPMBoundingBoxesAux.size(); i++){
            Rect Aux1 = DPMBoundingBoxesAux[i].rect;
            float score = DPMBoundingBoxesAux[i].score;
            NoNormalizedDPMScores.push_back(score);
            Camera.DPMBoundingBoxes.push_back(Aux1);
        }
    }

    // Score Normalization. Unity-based normalization
    double MaxDPMScore = 1.1371;
    double MinDPMScore = -5;

    for(int j = 0; j < NoNormalizedDPMScores.size(); j++){
        double Score = NoNormalizedDPMScores[j];
        double NormalizedScore = (Score - MinDPMScore) / (MaxDPMScore - MinDPMScore);
        Camera.DPMScores.push_back(NormalizedScore);
    }

    AllPedestrianVector = Camera.DPMBoundingBoxes;
    AllPedestrianVectorScore = Camera.DPMScores;
}

void PeopleDetector::paintBoundingBoxes(Mat &ActualFrame, string Method, vector<Rect> BoundingBoxes, int CameraNumber, int Thickness)
{
    Scalar Color;
    if (CameraNumber == 1)
        Color = Scalar (0, 255, 0);
    else if (CameraNumber == 2)
        Color = Scalar (255, 0, 0);
    else if (CameraNumber == 3)
        Color = Scalar (0, 0, 255);

    for (size_t i = 0; i < BoundingBoxes.size(); i++) {
        Rect r = BoundingBoxes[i];
        if (!Method.compare("HOG")) {
            // The HOG detector returns slightly larger rectangles than the real objects.
            // so we slightly shrink the rectangles to get a nicer output.
            r.x += cvRound(r.width*0.1);
            r.width = cvRound(r.width*0.8);
            r.y += cvRound(r.height*0.07);
            r.height = cvRound(r.height*0.8);
        }
        rectangle(ActualFrame, r.tl(), r.br(), Color, Thickness);
    }
}

void PeopleDetector::ACFPeopleDetection(CameraStream &Camera, bool PDFiltering)
{
    Camera.ACFBoundingBoxes.clear();
    Camera.ACFScores.clear();
    vector<double> NoNormalizedACFScores;

    // PEOPLE DETECTION FILTERING
    if (PDFiltering){
        if (Camera.FGImages.size() > 0){
            //cout << "Using BKG information" << endl;
            for (size_t i = 0; i < Camera.FGImages.size(); i++) {
                // Auxiliar ActualFrame
                Mat AuxiliarFrame = Camera.FGImages[i].clone();
                //Mat AuxiliarFrame2 = Camara.FGImages[i].clone();
                Rect Offset = Camera.FGBlobs[i];

                if (AuxiliarFrame.rows > 125 && AuxiliarFrame.cols > 125){
                    // Local detection vector
                    DetectionList ACFDetectionsList;
                    // ACF detector
                    ACFDetectionsList = ACFdetector.applyDetector(AuxiliarFrame);

                    //NMS
                    DetectionList ACFDetectionsListNMS;
                    NonMaximumSuppression NMS;
                    ACFDetectionsListNMS = NMS.dollarNMS(ACFDetectionsList);

                    // Convert from DetectionList to vector<Rect>
                    for (unsigned int i = 0; i < ACFDetectionsListNMS.Ds.size(); i++){
                        Detection Ds = ACFDetectionsListNMS.Ds[i];

                        Rect Rectangle;
                        Rectangle.x = Ds.m_x;
                        Rectangle.y = Ds.m_y;
                        Rectangle.width = Ds.m_width;
                        Rectangle.height = Ds.m_height;

                        float score = Ds.m_score;

                        // Convert top-left corner co-ordinates from small image to
                        // complete frame reference
                        Rectangle.x = Rectangle.x + Offset.x;
                        Rectangle.y = Rectangle.y + Offset.y;

                        NoNormalizedACFScores.push_back(score);
                        Camera.ACFBoundingBoxes.push_back(Rectangle);

                    }
                }
            }
        }
        else{
            // If there is no person on the semantic mask DPM does not search.
        }
    }
    // PEOPLE DETECTION WITHOUT FILTERING
    else{
        // Auxiliar ActualFrame
        Mat AuxiliarFrame = Camera.ActualFrame.clone();

        // Local detection vector
        DetectionList ACFDetectionsList;
        // ACF detector
        ACFDetectionsList = ACFdetector.applyDetector(AuxiliarFrame);

        //NMS
        DetectionList ACFDetectionsListNMS;
        NonMaximumSuppression NMS;
        ACFDetectionsListNMS = NMS.dollarNMS(ACFDetectionsList);

        // Convert from DetectionList to vector<Rect>
        for (unsigned int i = 0; i < ACFDetectionsListNMS.Ds.size(); i++){
            Detection Ds = ACFDetectionsListNMS.Ds[i];

            Rect Rectangle;
            Rectangle.x = Ds.m_x;
            Rectangle.y = Ds.m_y;
            Rectangle.width = Ds.m_width;
            Rectangle.height = Ds.m_height;

            float score = Ds.m_score;

            NoNormalizedACFScores.push_back(score);
            Camera.ACFBoundingBoxes.push_back(Rectangle);
        }
    }

    double MinACFScore = -15;
    double MaxACFScore = 107.46;

    // Score Normalization.
    for(int j = 0; j < NoNormalizedACFScores.size(); j++){
        double Score = NoNormalizedACFScores[j];
        double NormalizedScore = (Score - MinACFScore) / (MaxACFScore - MinACFScore);
        Camera.ACFScores.push_back(NormalizedScore);
    }

    AllPedestrianVector = Camera.ACFBoundingBoxes;
    AllPedestrianVectorScore = Camera.ACFScores;
}

void PeopleDetector::PSPNetScores(int CameraNumber, String FrameNumber)
{
    int FrameNumber2 = atoi(FrameNumber.c_str()) - 1;
    string SemScoresPath;

    if (FrameNumber2 < 10){
        // Add 000 to the path string
        SemScoresPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Scores PSP/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + "000" + to_string(FrameNumber2) + ".png";
    }
    else if (FrameNumber2 < 100){
        // Add 00 to the path string
        SemScoresPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Scores PSP/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + "00" + to_string(FrameNumber2) + ".png";
    }
    else if (FrameNumber2 < 1000){
        // Add 0 to the path string
        SemScoresPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Scores PSP/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + "0" + to_string(FrameNumber2) + ".png";
    }
    else{
        SemScoresPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Scores PSP/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + to_string(FrameNumber2) + ".png";
    }

    Mat ScoresImage = imread(SemScoresPath);

    vector<double> NoNormalizedScores, NormalizedScores;

    for(int i = 0; i < AllPedestrianVector.size(); i++){
        Rect Blob = AllPedestrianVector[i];
        Mat Aux = ScoresImage(Blob);

        int rows = Aux.rows;
        int cols = Aux.cols;

        double s = 0;
        int counter = 0;

        for(int row = 0; row < rows; row++){
            for(int col = 0; col < cols; col++){
                double pixel = Aux.at<uchar>(row, col);
                if(pixel > 10){
                    s = s + pixel;
                    counter++;
                }
            }
        }

        if(counter != 0){
            s = s / counter;
            s = s / 100;
            NoNormalizedScores.push_back(s);
        }
        else{
            s = 0;
            NoNormalizedScores.push_back(s);
        }

    }

    // Score Normalization
    double MaxPSPScore = 0.9407;
    double MinPSPScore = 0;

    for(int j = 0; j < NoNormalizedScores.size(); j++){
        double Score = NoNormalizedScores[j];
        double NormalizedScore = (Score - MinPSPScore) / (MaxPSPScore - MinPSPScore);
        NormalizedScores.push_back(NormalizedScore);
    }

    AllPedestrianVectorScore = NormalizedScores;
}

void PeopleDetector::ThresholdDetections(vector<Rect> Detections,  vector<double> Scores, double Threshold)
{
    if(Detections.empty())
        return;

    vector<int> SuppresedBlobs;

    for(int Index = 0; Index < Scores.size(); Index++){
        double Score = Scores[Index];
        if(Score < Threshold)
            SuppresedBlobs.push_back(Index);
    }

    sort(SuppresedBlobs.begin(), SuppresedBlobs.end(), greater<int>());

    for(size_t i = 0; i < SuppresedBlobs.size(); i++){
        int Index = SuppresedBlobs[i];
        Detections.erase(Detections.begin() + Index);
        Scores.erase(Scores.begin() + Index);
    }

    AllPedestrianVector.clear();
    AllPedestrianVectorScore.clear();
    AllPedestrianVector = Detections;
    AllPedestrianVectorScore = Scores;
}

void PeopleDetector::projectBlobs(vector<Rect> BoundingBoxes, vector<double> scores, Mat Homography, Mat HomographyBetweenViews, Mat &CenitalPlane, int CameraNumber, String RepresentationOption)
{
    if (BoundingBoxes.empty()){
        ProjectedLeftPoints.clear();
        ProjectedRightPoints.clear();
        ProjectedCenterPoints.clear();
        return;
    }

    Mat Gaussian;
    double score;
    Scalar SColor;
    Mat X, Y;

    if (CameraNumber == 1)
        SColor = Scalar (0, 255, 0);
    else if (CameraNumber == 2)
        SColor = Scalar (255, 0, 0);
    else if (CameraNumber == 3)
        SColor = Scalar (0, 0, 255);

    vector<Point2f> LeftCornerVectors, RightCornerVectors;
    ProjectedLeftPoints.clear();
    ProjectedRightPoints.clear();

    // Extract bottom bounding box segment
    for (size_t i = 0; i < BoundingBoxes.size(); i++) {
        // Extract the corresponding rectangle
        Rect r = BoundingBoxes[i];
        Point2f LeftCorner, RightCorner;

        // Extract Coordinates of the bottom segment
        LeftCorner.x = cvRound(r.x);
        LeftCorner.y = cvRound(r.y + r.height);
        RightCorner.x = cvRound(r.x + r.width);
        RightCorner.y = cvRound(r.y + r.height);

        // Same coordinates in vectors
        LeftCornerVectors.push_back(LeftCorner);
        RightCornerVectors.push_back(RightCorner);
    }
    // Apply Homography to vectors of Points to find the projection in the view
    perspectiveTransform(LeftCornerVectors, ProjectedLeftPoints, HomographyBetweenViews);
    perspectiveTransform(RightCornerVectors, ProjectedRightPoints, HomographyBetweenViews);
    // Apply Homography to vectors of Points to find the projection in the cenital plane
    perspectiveTransform(ProjectedLeftPoints, ProjectedLeftPoints, Homography);
    perspectiveTransform(ProjectedRightPoints, ProjectedRightPoints, Homography);

    // Vector to save the coordinates of projected squares for gaussians
    ProjectedCenterPoints.clear();

    for (size_t i = 0; i < ProjectedLeftPoints.size(); i++) {
        // Left Projected Point
        Point2f LeftProjected = ProjectedLeftPoints[i];
        // Rigth Projected Point
        Point2f RightProjected = ProjectedRightPoints[i];

        // Middle Segment Point
        Point2f MiddleSegmentPoint;
        MiddleSegmentPoint.x = cvRound((RightProjected.x + LeftProjected.x) / 2);
        MiddleSegmentPoint.y = cvRound((RightProjected.y + LeftProjected.y) / 2);

        // Direction Vector From Left Point to Rigth Point
        Point2f VectorLeft2Rigth;
        VectorLeft2Rigth.x = LeftProjected.x - RightProjected.x;
        VectorLeft2Rigth.y = LeftProjected.y - RightProjected.y;

        // Normalize Direction Vector
        float mag = sqrt (VectorLeft2Rigth.x * VectorLeft2Rigth.x + VectorLeft2Rigth.y * VectorLeft2Rigth.y);
        VectorLeft2Rigth.x = VectorLeft2Rigth.x / mag;
        VectorLeft2Rigth.y = VectorLeft2Rigth.y / mag;

        Point2f C;
        Scalar PerpendicularColor;
        float length;

        // Depending on the camera the direction of the perpendicular line is
        // different
        if(CameraNumber == 1){
            // Rotate direction vector 90º
            float temp = VectorLeft2Rigth.x;
            VectorLeft2Rigth.x = - VectorLeft2Rigth.y;
            VectorLeft2Rigth.y = temp;

            // Length of the new perpedicular line
            length = (RightProjected.x - LeftProjected.x)/2;
            PerpendicularColor = Scalar(255, 0, 255);
        }
        if(CameraNumber == 2){
            // Rotate direction vector 90º
            float temp = VectorLeft2Rigth.x;
            VectorLeft2Rigth.x = VectorLeft2Rigth.y;
            VectorLeft2Rigth.y = temp;

            // Length of the new perpedicular line
            length = (RightProjected.y - LeftProjected.y)/2;
            PerpendicularColor = Scalar(0, 255, 255);
        }
        if(CameraNumber == 3){
            // Rotate direction vector 90º
            float temp = VectorLeft2Rigth.x;
            VectorLeft2Rigth.x = - VectorLeft2Rigth.y;
            VectorLeft2Rigth.y = temp;

            // Length of the new perpedicular line
            length = (RightProjected.y - LeftProjected.y)/2;
            PerpendicularColor = Scalar(255, 255, 255);
        }

        // Center of the projected square
        C.x = MiddleSegmentPoint.x + VectorLeft2Rigth.x * length;
        C.y = MiddleSegmentPoint.y + VectorLeft2Rigth.y * length;

        // Save projected square central point
        ProjectedCenterPoints.push_back(C);

        if (!RepresentationOption.compare("Lines")){
            // Projection Line
            line(CenitalPlane, LeftProjected, RightProjected, SColor, 4);
            // Perpendicular line. New line at C pointing direction of Direction Vector
            line(CenitalPlane, MiddleSegmentPoint, C, PerpendicularColor, 4);
        }
        else if (!RepresentationOption.compare("Gaussians")){
            // Mesgrid function
            meshgrid(X, Y, CenitalPlane.rows, CenitalPlane.cols);

            CenitalPlane.convertTo(CenitalPlane, CV_32FC3, 1/255.0);

            if (!scores.empty()) {
                // Extract the maximum score from the vector
                double MaxScore = *max_element(scores.begin(), scores.end());
                if (MaxScore > 1){
                    score = ((exp(-(scores[i]/MaxScore)))/0.02) - 15;
                }
                else {
                    score = ((exp(-scores[i]))/0.02) - 15;
                }
            }
            else {
                score = 5;
            }

            // Draw a Gaussian of mean = C and std = score
            gaussianFunction(Gaussian, X, Y, C, score, CameraNumber);
            // Add gaussian to CenitalPlane to display
            add(Gaussian, CenitalPlane, CenitalPlane);

            CenitalPlane.convertTo(CenitalPlane, CV_8UC3, 255.0);
        }
    }
}

void PeopleDetector::meshgrid(Mat &X, Mat &Y, int rows, int cols)
{
    X = Mat::zeros(1, cols, CV_32FC1);
    Y = Mat::zeros(rows, 1, CV_32FC1);

    // Create incrementing row and column vector
    for (int i = 0; i < cols; i++)
        X.at<float>(0,i) = i;

    for (int i = 0; i < rows; i++)
        Y.at<float>(i,0) = i;

    // Create matrix repiting row and column
    X = repeat(X, rows, 1);
    Y = repeat(Y, 1, cols);
}

void PeopleDetector::gaussianFunction(Mat &Gaussian3C, Mat X, Mat Y, Point2f center, double score, int CameraNumber)
{
    Mat Gaussian;
    Mat Fra1, Fra2, Powx1, Powx2, Powy1, Powy2;
    double A = 1;
    double MeanX, MeanY, sigmaX, sigmaY;

    // Gaussian Parameters
    MeanX = center.x;
    MeanY = center.y;
    sigmaX = score;
    sigmaY = score;

    // X Equation
    pow((X - MeanX), 2, Powx1);
    pow(sigmaX, 2, Powx2);
    Powx2 = 2*Powx2;
    divide(Powx1, Powx2, Fra1);

    // Y Equation
    pow((Y - MeanY), 2, Powy1);
    pow(sigmaY, 2, Powy2);
    Powy2 = 2*Powy2;
    divide(Powy1, Powy2, Fra2);

    // Combine X and Y fractions
    Gaussian = -(Fra1 + Fra2);
    exp(Gaussian, Gaussian);
    Gaussian = A*Gaussian;

    // Convert Gaussian to 3-channel matrix
    vector<cv::Mat> GaussianChannels(3);
    GaussianChannels.at(0) = Mat::zeros(X.rows, X.cols, CV_32FC1);
    GaussianChannels.at(1) = Mat::zeros(X.rows, X.cols, CV_32FC1);
    GaussianChannels.at(2) = Mat::zeros(X.rows, X.cols, CV_32FC1);

    if (CameraNumber == 1)
        GaussianChannels.at(1) = Gaussian;

    if (CameraNumber == 2)
        GaussianChannels.at(0) = Gaussian;

    if (CameraNumber == 3)
        GaussianChannels.at(2) = Gaussian;

    merge(GaussianChannels, Gaussian3C);
}

void PeopleDetector::ReprojectionFusion(vector<Point2f> ProjCenterPoints, vector<Point2f> ProjLeftPoints, vector<Point2f> ProjRightPoints, Mat Homography, Mat HomographyBetweenViews, Mat &ActualFrame)
{
    if ((ProjCenterPoints.empty()) || (ProjLeftPoints.empty()) || (ProjRightPoints.empty())){
        return;
    }

    vector<Point2f> CenterPoints, LeftPoints, RightPoints;
    // Apply Homography inverse to reproject from the cneital plane to the view
    perspectiveTransform(ProjCenterPoints, CenterPoints, Homography.inv(DECOMP_LU));
    perspectiveTransform(ProjLeftPoints, LeftPoints, Homography.inv(DECOMP_LU));
    perspectiveTransform(ProjRightPoints, RightPoints, Homography.inv(DECOMP_LU));

    // Apply HomographyBetweenViews inverse to project from the view to the actual frame
    perspectiveTransform(CenterPoints, CenterPoints, HomographyBetweenViews.inv(DECOMP_LU));
    perspectiveTransform(LeftPoints, LeftPoints, HomographyBetweenViews.inv(DECOMP_LU));
    perspectiveTransform(RightPoints, RightPoints, HomographyBetweenViews.inv(DECOMP_LU));

    // Draw into the actual frame the reprojected detections
    for(int n = 0; n < CenterPoints.size(); n++){
        Point2f Center = CenterPoints[n];
        Point2f LeftCorner = LeftPoints[n];
        Point2f RightCorner = RightPoints[n];
        Rect Blob;

        Blob.width = (RightCorner.x - LeftCorner.x) + 50;
        Blob.height = Blob.width * 4;
        Blob.x = RightCorner.x - Blob.width;
        Blob.y = LeftCorner.y - Blob.height;

        AllPedestrianVector.push_back(Blob);
        circle(ActualFrame, Center, 10, Scalar(255,255,255), 2);
    }
}

void PeopleDetector::SemanticConstraining(vector<Rect> &AllPedestrianVector, int CameraNumber, Mat &ActualFrame, Mat Homography, Mat HomographyBetweenViews)
{
    Mat CommonImage1, CommonImage2;

    // Load the common images depending on the camera number
    if(CameraNumber == 1) {
        CommonImage1 = imread("/Users/alex/Desktop/CommonSemantic13.png", CV_LOAD_IMAGE_GRAYSCALE);
        CommonImage2 = imread("/Users/alex/Desktop/CommonSemantic12.png", CV_LOAD_IMAGE_GRAYSCALE);
    }
    else if(CameraNumber == 2) {
        CommonImage1 = imread("/Users/alex/Desktop/CommonSemantic12.png", CV_LOAD_IMAGE_GRAYSCALE);
        CommonImage2 = imread("/Users/alex/Desktop/CommonSemantic23.png", CV_LOAD_IMAGE_GRAYSCALE);
    }
    else if(CameraNumber == 3) {
        CommonImage1 = imread("/Users/alex/Desktop/CommonSemantic13.png", CV_LOAD_IMAGE_GRAYSCALE);
        CommonImage2 = imread("/Users/alex/Desktop/CommonSemantic23.png", CV_LOAD_IMAGE_GRAYSCALE);
    }

    if((! CommonImage1.data) || (! CommonImage2.data)) {
        cout <<  "Could not open the common semantic images for SemanticConstraining function" << endl ;
        return;
    }

    // Vector for supressed points due to semantic constrains
    SupressedIndices.clear();
    int Counter = 0;

    for(int i = 0; i < AllPedestrianVector.size(); i++){
        Rect Blob = AllPedestrianVector[i];
        Point2f BottomCenter, BottomCenterProjected;
        vector<Point2f> VectorAux;

        // COmpute bottom middle part of the blob and save it in an auxiliar vector
        BottomCenter.x = cvRound(Blob.x + Blob.width/2);
        BottomCenter.y = Blob.y + Blob.height;
        VectorAux.push_back(BottomCenter);

        // Transform the point to the cenital plane
        perspectiveTransform(VectorAux, VectorAux, HomographyBetweenViews);
        perspectiveTransform(VectorAux, VectorAux, Homography);

        // Extract the projected point
        BottomCenterProjected = VectorAux[0];

        if((BottomCenterProjected.x > 0) && (BottomCenterProjected.y > 0) && (BottomCenterProjected.x < CommonImage1.cols) && (BottomCenterProjected.y < CommonImage1.rows)) {
            int Label1 = CommonImage1.at<uchar>(cvRound(BottomCenterProjected.y), cvRound(BottomCenterProjected.x)) / 20;
            int Label2 = CommonImage2.at<uchar>(cvRound(BottomCenterProjected.y), cvRound(BottomCenterProjected.x)) / 20;

            if (!(Label1 == 3 || Label2 == 3)){
                SupressedIndices.push_back(Counter);
                putText(ActualFrame, "BLOB SUPRESSED", BottomCenter, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1.5);
            }
        }
        else{
            SupressedIndices.push_back(Counter);
            putText(ActualFrame, "BLOB SUPRESSED", BottomCenter, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1.5);
        }
        Counter++;
    }

    sort(SupressedIndices.begin(), SupressedIndices.end(), greater<int>());

    // Suppress the blobs from the vector
    for(int k = 0; k < SupressedIndices.size(); k++){
        int Index = SupressedIndices[k];
        AllPedestrianVector.erase(AllPedestrianVector.begin() + Index);
    }
}

void PeopleDetector::ExtractDataUsage(int CameraNumber, String FrameNumber, Mat Homography, Mat HomographyBetweenViews)
{
    if(AllPedestrianVectorNMS.empty() || Homography.empty() || HomographyBetweenViews.empty()){
        StatisticalBlobFile << "0" << endl;
        return;
    }

    Mat StatisticalMap = imread("/Users/alex/Desktop/StatisticalMap.png", CV_LOAD_IMAGE_GRAYSCALE);

    int FloorCount = 0, DoorCount = 0, ChairCount = 0;

    for(int i = 0; i < AllPedestrianVectorNMS.size(); i++){
        Rect Blob = AllPedestrianVectorNMS[i];
        int MapLabel;
        Point2f BottomCenter, BottomCenterProjected;
        vector<Point2f> VectorAux;

        // COmpute bottom middle part of the blob and save it in an auxiliar vector
        BottomCenter.x = cvRound(Blob.x + Blob.width/2);
        BottomCenter.y = Blob.y + Blob.height;
        VectorAux.push_back(BottomCenter);

        // Transform the point to the cenital plane
        perspectiveTransform(VectorAux, VectorAux, HomographyBetweenViews);
        perspectiveTransform(VectorAux, VectorAux, Homography);

        // Extract the projected point
        BottomCenterProjected = VectorAux[0];

        StatisticalBlobFile << BottomCenterProjected.x << " " << BottomCenterProjected.y << " ";

        if((BottomCenterProjected.x > 0) && (BottomCenterProjected.y > 0) && (BottomCenterProjected.x < StatisticalMap.cols) && (BottomCenterProjected.y < StatisticalMap.rows)) {
            // Check label of Statistical Map
            MapLabel = StatisticalMap.at<uchar>(cvRound(BottomCenterProjected.y), cvRound(BottomCenterProjected.x));

            //cout << "Pos: " << BottomCenterProjected << " with label " << to_string(MapLabel) << endl;
            // If floor, door or chair sum 1
            if(MapLabel == 3){
                FloorCount++;
            }
            else if(MapLabel == 8){
                DoorCount++;
            }
            else if(MapLabel == 9){
                ChairCount++;
            }
        }
    }
    StatisticalBlobFile << endl;
    cout << "Frame: " << FrameNumber << ". Camera " << to_string(CameraNumber) << ". " << to_string(FloorCount) << " on the floor. "
         << to_string(DoorCount) << " usign the doors. " << to_string(ChairCount) << " seated on chairs." << endl;
}
