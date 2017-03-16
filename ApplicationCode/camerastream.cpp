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

CameraStream::CameraStream(){}
CameraStream::~CameraStream(){}

using namespace cv;
using namespace std;

void CameraStream::VideoOpenning(string InputPath)
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

void CameraStream::FastRCNNPeopleDetection(string FrameNumber, string Method)
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

void CameraStream::decodeBlobFile(string FileName, string FrameNumber)
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

void CameraStream::maskEnhancement(Mat BackgroundMask)
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

void CameraStream::imageEnhancement()
{
    // Increase video size
    cv::resize(ActualFrame, ActualFrame, {ActualFrame.cols*2, ActualFrame.rows*2}, INTER_LANCZOS4);
    // Remove interpolation artifacts by a low-pass filtering
    cv::GaussianBlur(ActualFrame, ActualFrame, Size(1,1), 15);

    Width = ActualFrame.cols;
    Height = ActualFrame.rows;
}

void CameraStream::computeHomography()
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

void CameraStream::saveWarpImages(Mat ActualFrame, Mat Homography, String FrameNumber)
{
    // Extract image warping
    Mat ImageWarping;
    ImageWarping = Mat::zeros(600, 1500, CV_64F);
    warpPerspective(ActualFrame, ImageWarping, Homography, ImageWarping.size());
    //String WindowName = "Wrapped Camera " + to_string(CameraNumber);
    //imshow(WindowName, ImageWarping);

    String ImageName = "/Users/alex/IPCV-MasterThesis/ApplicationCode/Wrapped Images/Camera " + to_string(CameraNumber) + "/Frame" + FrameNumber + ".png";

    imwrite(ImageName, ImageWarping);
}

void CameraStream::projectSemantic(Mat &CenitalPlane)
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
    CenitalPath = "/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/CenitalViewRombo.png";
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
