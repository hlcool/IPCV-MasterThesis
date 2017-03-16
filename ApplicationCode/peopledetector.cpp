#include "peopledetector.h"
#include "camerastream.h"
#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <boost/lexical_cast.hpp>

PeopleDetector::PeopleDetector(){}
PeopleDetector::~PeopleDetector(){}

void PeopleDetector::MainPeopleDetection(CameraStream &Camera1, CameraStream &Camera2, CameraStream &Camera3, String CBOption, Mat &CenitalPlane)
{
    if (!CBOption.compare("HOG")){
        // HOG Detector

        // Camera 1
        HOGPeopleDetection(Camera1);
        paintBoundingBoxes(Camera1.ActualFrame, CBOption, Camera1.HOGBoundingBoxesNMS, Scalar (0, 255, 0), 1);
        projectBlobs(Camera1.HOGBoundingBoxesNMS, Camera1.HOGScores, Camera1.Homography, "GREEN", CenitalPlane, Camera1.CameraNumber);

        // Camera 2
        HOGPeopleDetection(Camera2);
        paintBoundingBoxes(Camera2.ActualFrame, CBOption, Camera2.HOGBoundingBoxesNMS, Scalar (255, 0, 0), 1);
        projectBlobs(Camera2.HOGBoundingBoxesNMS, Camera2.HOGScores, Camera2.Homography, "BLUE", CenitalPlane, Camera2.CameraNumber);

        // Camera 3
        HOGPeopleDetection(Camera3);
        paintBoundingBoxes(Camera3.ActualFrame, CBOption, Camera3.HOGBoundingBoxesNMS, Scalar (0, 0, 255), 1);
        projectBlobs(Camera3.HOGBoundingBoxesNMS, Camera3.HOGScores, Camera3.Homography, "RED", CenitalPlane, Camera3.CameraNumber);
    }
    else if(!CBOption.compare("FastRCNN")){
        // FastRCNN Detector

        //FastRCNNPeopleDetection(FrameNumber, Camera1.FastRCNNMethod);
        //paintBoundingBoxes(Camera1.ActualFrame, CBOption, Camera1.RCNNBoundingBoxesNMS, Scalar (0, 0, 255), 1);
        //projectBlobs(Camera1.RCNNBoundingBoxesNMS, Camera1.RCNNScores, Camera1.Homography, "RED", CenitalPlane);
    }
    else if(!CBOption.compare("DPM")){
        // DPM Detector

        // Camera 1
        DPMPeopleDetection(Camera1);
        paintBoundingBoxes(Camera1.ActualFrame, CBOption, Camera1.DPMBoundingBoxes, Scalar (0, 255, 0), 1);
        projectBlobs(Camera1.DPMBoundingBoxes, Camera1.DPMScores, Camera1.Homography, "GREEN", CenitalPlane, Camera1.CameraNumber);

        // Camera 2
        DPMPeopleDetection(Camera2);
        paintBoundingBoxes(Camera2.ActualFrame, CBOption, Camera2.DPMBoundingBoxes, Scalar (255, 0, 0), 1);
        projectBlobs(Camera2.DPMBoundingBoxes, Camera2.DPMScores, Camera2.Homography, "BLUE", CenitalPlane, Camera2.CameraNumber);

        // Camera 3
        DPMPeopleDetection(Camera3);
        paintBoundingBoxes(Camera3.ActualFrame, CBOption, Camera3.DPMBoundingBoxes, Scalar (0, 0, 255), 1);
        projectBlobs(Camera3.DPMBoundingBoxes, Camera3.DPMScores, Camera3.Homography, "RED", CenitalPlane, Camera3.CameraNumber);

    }
}

void PeopleDetector::HOGPeopleDetection(CameraStream &Camara)
{
    // Clear vectors
    Camara.HOGBoundingBoxes.clear();
    Camara.HOGBoundingBoxesNMS.clear();

    // Initialice the SVM
    HOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    // HOG Detector
    HOG.detectMultiScale(Camara.ActualFrame, Camara.HOGBoundingBoxes, Camara.HOGScores, 0, Size(8, 8), Size(32, 32), 1.1, 2);
    Camara.HOGBoundingBoxesNMS = Camara.HOGBoundingBoxes;
    //non_max_suppresion(HOGBoundingBoxes, HOGBoundingBoxesNMS, 0.65);
}

void PeopleDetector::DPMPeopleDetection(CameraStream &Camara)
{
    Camara.DPMBoundingBoxes.clear();
    // Auxiliar ActualFrame
    Mat AuxiliarFrame = Camara.ActualFrame.clone();

    // Local detection vector
    vector<DPMDetector::ObjectDetection> DPMBoundingBoxesAux;
    // DPM detector with NMS
    DPMdetector->detect(AuxiliarFrame, DPMBoundingBoxesAux);

    // Convert from vector<ObjectDetection> to vector<Rect>
    for (unsigned int i = 0; i < DPMBoundingBoxesAux.size(); i++){
        Rect Aux1 = DPMBoundingBoxesAux[i].rect;
        float score = DPMBoundingBoxesAux[i].score;
        Camara.DPMScores.push_back(score);
        Camara.DPMBoundingBoxes.push_back(Aux1);
    }
}

void PeopleDetector::paintBoundingBoxes(Mat &ActualFrame, string Method, vector<Rect> BoundingBoxes, Scalar Color, int Thickness)
{
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

void PeopleDetector::projectBlobs(vector<Rect> BoundingBoxes, vector<double> scores, Mat Homography, string Color, Mat &CenitalPlane, int CameraNumber)
{
    if (BoundingBoxes.empty())
        return;

    Mat Gaussian;
    double score;
    Scalar SColor;
    Mat X, Y;

    if (!Color.compare("RED"))
        SColor = Scalar(0, 0, 255);
    else if (!Color.compare("GREEN"))
        SColor = Scalar(0, 255, 0);
    else if (!Color.compare("BLUE"))
        SColor = Scalar(255, 0, 0);

    vector<Point2f> AuxPointVector;
    vector<Point2f> ProjectedPoints;

    // Extract bottom midle bounding box point
    for (size_t i = 0; i < BoundingBoxes.size(); i++) {
        // Extract the corresponding rectangle
        Rect r = BoundingBoxes[i];
        Point2f Point;

        // Bottom midle point of the blob
        Point.x = cvRound(r.x + r.width/2);
        Point.y = cvRound(r.y + r.height);

        // Store project Point in vector
        AuxPointVector.push_back(Point);
    }

    // Apply Homography to vector of Points to find the projection
    perspectiveTransform(AuxPointVector, ProjectedPoints, Homography);

    // Convert CenitalPlane to flaoting point mat to add the Gaussians
    //CenitalPlane.convertTo(CenitalPlane, CV_32FC3, 1/255.0);

    // Mesgrid function
    meshgrid(X, Y, CenitalPlane.rows, CenitalPlane.cols);

    // Extract the maximum score from the vector
    double MaxScore = *max_element(scores.begin(), scores.end());

    // Extract projected points and create Gaussians
    for (size_t i = 0; i < ProjectedPoints.size(); i++) {
        Point2f center = ProjectedPoints[i];
        if (!scores.empty()) {
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

        // Draw a Gaussian of mean = center and std = score
        gaussianFunction(Gaussian, X, Y, center, score, CameraNumber);

        // Add gaussian to CenitalPlane to display
        add(Gaussian, CenitalPlane, CenitalPlane);
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

void PeopleDetector::non_max_suppresion(const vector<Rect> &srcRects, vector<Rect> &resRects, float thresh)
{
    resRects.clear();

    const size_t size = srcRects.size();
    if (!size)
        return;

    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i) {
        idxs.insert(pair<int, size_t>(srcRects[i].br().y, i));
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0) {
        // grab the last rectangle
        auto lastElem = --end(idxs);
        const Rect& rect1 = srcRects[lastElem->second];

        resRects.push_back(rect1);

        idxs.erase(lastElem);

        for (auto pos = begin(idxs); pos != end(idxs); ) {
            // grab the current rectangle
            const Rect& rect2 = srcRects[pos->second];

            float intArea = (rect1 & rect2).area();
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;

            cout << overlap << endl;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh) {
                pos = idxs.erase(pos);
            }
            else {
                ++pos;
            }
        }
    }
}
