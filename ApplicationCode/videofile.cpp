#include "videofile.h"
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

VideoFile::VideoFile(){}
VideoFile::~VideoFile(){}

using namespace cv;
using namespace std;

void VideoFile::VideoOpenning(string InputPath)
{
    // Open the videofile to check if it exists
    cap.open(InputPath);
    if (!cap.isOpened()) {
        cout << "Could not open video file " << InputPath << endl;
        exit(EXIT_FAILURE);
    }

    // Extract information from VideoCapture
    Width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    Height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    FrameRate = cap.get(CV_CAP_PROP_FPS);
    FrameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);

    // Display information
    cout << "Camera " << CameraNumber << " opened correctly"  << endl;
    cout << "The video to process has the following information:" << endl;
    cout << "Width: " << Width << ". Heigth: " << Height << ". Frames/second: " << FrameRate << endl;
    cout << "The total number of frames is: " << FrameNumber << endl;
    cout << "" << endl;
}

void VideoFile::HOGPeopleDetection(Mat ActualFrame)
{
    // Clear vectors
    HOGBoundingBoxes.clear();
    HOGBoundingBoxesNMS.clear();

    // Initialice the SVM
    HOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    // HOG Detector
    HOG.detectMultiScale(ActualFrame, HOGBoundingBoxes, HOGScores, 0, Size(8, 8), Size(32, 32), 1.1, 2);
    HOGBoundingBoxesNMS = HOGBoundingBoxes;
    //non_max_suppresion(HOGBoundingBoxes, HOGBoundingBoxesNMS, 0.65);
}

void VideoFile::FastRCNNPeopleDetection(string FrameNumber, string Method)
{
    // Clear vectors
    RCNNBoundingBoxes.clear();
    RCNNBoundingBoxesNMS.clear();
    RCNNScores.clear();

    // Decode de txt file for the desired frame number
    size_t slash = InputPath.find_last_of("/");
    size_t point = InputPath.find_last_of(".");
    string FileName = InputPath.substr(slash + 1, point - slash - 1);

    if (!Method.compare("fast"))
        FileName = "/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/People Detection/" + FileName + "fast.txt";
    else if (!Method.compare("accurate"))
        FileName = "/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/People Detection/" + FileName + "Accurate.txt";

    decodeBlobFile(FileName, FrameNumber);

    // Score average
    double average = accumulate( RCNNScores.begin(), RCNNScores.end(), 0.0) / RCNNScores.size();

    // Filter blobs by average
    for (size_t i = 0; i < RCNNBoundingBoxes.size(); i++) {
        if (RCNNScores[i] <= (average - (average * 0.05)) ) {
            RCNNBoundingBoxes.erase(RCNNBoundingBoxes.begin() + i);
            RCNNScores.erase(RCNNScores.begin() + i);
        }
    }
    RCNNBoundingBoxesNMS = RCNNBoundingBoxes;
    //non_max_suppresion(RCNNBoundingBoxes, RCNNBoundingBoxesNMS, 0.65);
}

void VideoFile::decodeBlobFile(string FileName, string FrameNumber)
{
    ifstream input(FileName);

    if (!input) {
        // The file does not exists
        cout << "The file containing the FastRCNN blobs does not exist" << endl;
        exit(EXIT_FAILURE);
    }

    // Auxiliary variables to store the information
    string AuxString;
    int x2, y2;
    double Score;
    Rect RectAux;
    size_t found;
    int Counter = 0;
    int LineCounter = 0;

    // Start decoding the file
    while (input >> AuxString){

        if (AuxString.find("Frame") != std::string::npos) {
            // Check if the desired line has been read and so
            // exit the function
            if (LineCounter == atoi(FrameNumber.c_str()))
                return;

            LineCounter++;
        }

        if (LineCounter == atoi(FrameNumber.c_str())) {
            //cout << "Line Counter " << LineCounter << ". Frame number: " << atoi(FrameNumber.c_str()) << endl;
            switch(Counter)
            {
            case 0:
                Counter++;
                break;
            case 1:
                // Case for x1
                found = AuxString.find(',');
                AuxString = AuxString.substr(1, found - 1 );
                RectAux.x = atoi(AuxString.c_str());
                Counter++;
                break;
            case 2:
                // Case for y1
                found = AuxString.find(',');
                AuxString = AuxString.substr(0, found);
                RectAux.y = atoi(AuxString.c_str());
                Counter++;
                break;
            case 3:
                // Case for x2
                found = AuxString.find(',');
                AuxString = AuxString.substr(0, found);
                x2 = atoi(AuxString.c_str());
                Counter++;
                break;
            case 4:
                // Case for y2
                found = AuxString.find(']');
                AuxString = AuxString.substr(0, found);
                y2 = atoi(AuxString.c_str());
                Counter++;
                break;
            case 5:
                // Case for "Score:"
                Counter++;
                break;
            case 6:
                // Case for score
                Score = boost::lexical_cast<double>(AuxString);

                // Save blob information into class variables
                RectAux.width = x2 - RectAux.x;
                RectAux.height = y2 - RectAux.y;
                RCNNBoundingBoxes.push_back(RectAux);
                RCNNScores.push_back(Score);

                // Restart the couter to read another blob
                Counter = 1;
                break;

            }
        }
    }
}

void VideoFile::DPMPeopleDetection(Mat ActualFrame)
{
    DPMBoundingBoxes.clear();

    // Local detection vector
    vector<DPMDetector::ObjectDetection> DPMBoundingBoxesAux;
    // DPM detector with NMS
    DPMdetector->detect(ActualFrame, DPMBoundingBoxesAux);

    // Convert from vector<ObjectDetection> to vector<Rect>
    for (unsigned int i = 0; i < DPMBoundingBoxesAux.size(); i++){
        Rect Aux1 = DPMBoundingBoxesAux[i].rect;
        float score = DPMBoundingBoxesAux[i].score;
        DPMScores.push_back(score);
        DPMBoundingBoxes.push_back(Aux1);
    }
}

void VideoFile::non_max_suppresion(const vector<Rect> &srcRects, vector<Rect> &resRects, float thresh)
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

void VideoFile::paintBoundingBoxes(Mat ActualFrame, string Method, vector<Rect> BoundingBoxes, Scalar Color, int Thickness)
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

void VideoFile::maskEnhancement(Mat BackgroundMask)
{
    // Dilatation and Erosion kernels
    Mat kernel_di = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));
    Mat kernel_ero = getStructuringElement(MORPH_ELLIPSE, Size(2, 2), Point(-1, -1));

    // Remove shadows from the mask. Only foreground is saved
    threshold(BackgroundMask, BackgroundMask, 250, 255, THRESH_BINARY);

    // Opening morphological operation
    erode(BackgroundMask, BackgroundMask, kernel_ero, Point(-1, -1));
    dilate(BackgroundMask, BackgroundMask, kernel_di, Point(-1, -1));
}

void VideoFile::imageEnhancement()
{
    // Increase video size
    cv::resize(ActualFrame, ActualFrame, {ActualFrame.cols*2, ActualFrame.rows*2}, INTER_LANCZOS4);
    // Remove interpolation artifacts by a low-pass filtering
    cv::GaussianBlur(ActualFrame, ActualFrame, Size(1,1), 15);

    Width = ActualFrame.cols;
    Height = ActualFrame.rows;
    if (FlagCOUT == 1) {
        cout << "The video has been resize to " << "Width: " << Width << ". Heigth: " << Height << endl;
        FlagCOUT = 0;
    }
}

void VideoFile::computeHomography()
{
    vector<Point2f> pts_src, pts_dst;
    string XCoord, YCoord;

    // CAMERA FRAME POINTS
    string FileName = "/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/Camera" + to_string(CameraNumber) + "PtsSrcFile.txt";
    ifstream input(FileName);

    if (!input) {
        // The file does not exists
        cout << "The file that should contain homography points for Camera " + to_string(CameraNumber) + " Frame do not exist" << endl;
        exit(EXIT_FAILURE);
    }

    // Start decoding the file with src points
    while (input >> XCoord){
        input >> YCoord;
        Point2f pt;
        pt.x = atoi(XCoord.c_str());
        pt.y = atoi(YCoord.c_str());
        pts_src.push_back(pt);
    }

    // CENITAL FRAME POINTS
    FileName = "/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/Camera" + to_string(CameraNumber) + "PtsDstFile.txt";
    ifstream input2(FileName);

    if (!input2) {
        // The file does not exists
        cout << "The file that should contain homography points for Cenital Frame for camera " + to_string(CameraNumber) + " do not exist" << endl;
        exit(EXIT_FAILURE);
    }

    // Start decoding the file with dst points
    while (input2 >> XCoord){
        input2 >> YCoord;
        Point2f pt;
        pt.x = atoi(XCoord.c_str());
        pt.y = atoi(YCoord.c_str());
        pts_dst.push_back(pt);
    }

    if (pts_dst.size() != pts_src.size()){
        cout << "The number of homography points for Camera " + to_string(CameraNumber) + " is not the same in source and destiny" << endl;
        exit(EXIT_FAILURE);
    }

    // Calculate Homography
    Homography = findHomography(pts_src, pts_dst, CV_LMEDS);
}

void VideoFile::projectBlobs(vector<Rect> BoundingBoxes, vector<double> scores, Mat Homography, string Color, Mat &CenitalPlane)
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
                //cout << "Score: " << scores[i]/MaxScore;
                score = ((exp(-(scores[i]/MaxScore)))/0.02) - 15;
                //cout << ". STD: " << score << endl;
            }
            else {
                //cout << "Score: " << scores[i];
                score = ((exp(-scores[i]))/0.02) - 15;
                //cout << ". STD: " << score << endl;
            }
        }
        else {
            score = 5;
        }

        // Draw a Gaussian of mean = center and std = score
        gaussianFunction(Gaussian, X, Y, center, score);

        // Add gaussian to CenitalPlane to display
        add(Gaussian, CenitalPlane, CenitalPlane);
    }
}

void VideoFile::projectSemantic(Mat &CenitalPlane)
{
    // Four floor points
    Mat overlay;
    double alpha = 0.3;
    vector<Point2f> FloorPoints;
    vector<Point2f> ProjectedFloor;
    Scalar Color;

    if (CameraNumber == 1){
        // Floor points preselected (x,y) for Camera1
        FloorPoints.push_back(Point2f(1, 149));
        FloorPoints.push_back(Point2f(639, 148));
        FloorPoints.push_back(Point2f(637, 477));
        FloorPoints.push_back(Point2f(2, 477));
        Color = Scalar(0,255,0);
    }
    if (CameraNumber == 2){
        // Floor points preselected (x,y) for Camera2
        FloorPoints.push_back(Point2f(25, 82));
        FloorPoints.push_back(Point2f(617, 80));
        FloorPoints.push_back(Point2f(621, 459));
        FloorPoints.push_back(Point2f(25, 460));
        Color = Scalar(255,0,0);
    }
    if (CameraNumber == 3){
        // Floor points preselected (x,y) for Camera3
        FloorPoints.push_back(Point2f(15, 75));
        FloorPoints.push_back(Point2f(604, 75));
        FloorPoints.push_back(Point2f(633, 469));
        FloorPoints.push_back(Point2f(10, 467));
        Color = Scalar(0,0,255);
    }

    // Apply Homography to vector of Points to find the projection
    perspectiveTransform(FloorPoints, ProjectedFloor, Homography);

    // Convert vector of points into array of points
    Point ArrayProjectedPoints[4];
    copy(ProjectedFloor.begin(), ProjectedFloor.end(), ArrayProjectedPoints);

    // Clean previous cenital view
    string CenitalPath;
    CenitalPath = "/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/CenitalView.png";
    if (CameraNumber == 1){
        CenitalPlane = imread(CenitalPath);
    }

    if(!CenitalPlane.data ){
        cout <<  "Could not open or find the image " << CenitalPath << std::endl ;
        exit(EXIT_FAILURE);
    }

    // Copy the cenital image to an overlay
    CenitalPlane.copyTo(overlay);

    // Create the poligon and add transparency
    fillConvexPoly( overlay, ArrayProjectedPoints, 4, Color );
    addWeighted(overlay, alpha, CenitalPlane, 1 - alpha, 0, CenitalPlane);

    if (CameraNumber == 3){
        CenitalPlane.convertTo(CenitalPlane, CV_32FC3, 1/255.0);
    }
}

void VideoFile::meshgrid(Mat &X, Mat &Y, int rows, int cols)
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

void VideoFile::gaussianFunction(Mat &Gaussian3C, Mat X, Mat Y, Point2f center, double score)
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
