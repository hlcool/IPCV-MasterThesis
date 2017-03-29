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

    // Load Semantic images for the camera
    vector<size_t> characterLocations;
    for(size_t i =0; i < InputPath.size(); i++){
        if(InputPath[i] == '/')
            characterLocations.push_back(i);
    }

    size_t Pos = characterLocations[characterLocations.size() - 2];
    string ImagesPath = InputPath.substr(0, Pos);

    ImagesPath = ImagesPath + "/Semantic Images/Camera " + to_string(CameraNumber) + ".png";
    cout << ImagesPath << endl;
    SemanticImage = imread(ImagesPath);
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

    String ImageName = "/Users/alex/IPCV-MasterThesis/ApplicationCode/Wrapped Images/Camera " + to_string(CameraNumber) + "/Frame" + FrameNumber + ".png";
    imwrite(ImageName, ImageWarping);
}

void CameraStream::ProjectFloorPoints()
{
    // Extract floor mask
    Mat FloorMask;
    Mat SemanticImageGray;

    cvtColor(SemanticImage, SemanticImageGray , CV_BGR2GRAY);
    compare(SemanticImageGray, 3, FloorMask, CMP_EQ);

    vector<Point2f> FloorPoints;
    vector<Point2f> ProjectedFloor;

    // output, locations of non-zero pixels
    for (int i = 0; i < FloorMask.cols; i++ ) {
            for (int j = 0; j < FloorMask.rows; j++) {
                if (FloorMask.at<uchar>(j, i) == 255) {
                    //cout << i << ", " << j << endl;
                    FloorPoints.push_back(Point2f(j, i));
                }
            }
        }

    // Apply Homography to vector of Points to find the projection
    perspectiveTransform(FloorPoints, ProjectedFloor, Homography);

    // Convert vector of points into array of points
    NumberFloorPoints = static_cast<int>(FloorPoints.size());
    ArrayProjectedFloorPoints = new Point[NumberFloorPoints];
    copy(ProjectedFloor.begin(), ProjectedFloor.end(), ArrayProjectedFloorPoints);
}

void CameraStream::drawSemantic(Mat &CenitalPlane)
{
    Mat overlay;
    double alpha = 0.3;
    Scalar Color;

    if (CameraNumber == 1)
        Color = Scalar(0,255,0);
    if (CameraNumber == 2)
        Color = Scalar(255,0,0);
    if (CameraNumber == 3)
        Color = Scalar(0,0,255);

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
    fillConvexPoly(overlay, ArrayProjectedFloorPoints, NumberFloorPoints, Color );
    addWeighted(overlay, alpha, CenitalPlane, 1 - alpha, 0, CenitalPlane);

    if (CameraNumber == 3){
        CenitalPlane.convertTo(CenitalPlane, CV_32FC3, 1/255.0);
    }
}

void CameraStream::extractFGBlobs(Mat fgmask)
{
    // Required variables for connected component analysis
    Point pt;
    Rect RectangleOutput;
    Scalar NewValue = 254;
    Scalar MaxMin = 1;
    int Flag = 8;

    // Clear blob list (to fill with this function)
    vector<Rect> bloblist;
    vector<Rect> bloblist_joined;

    bloblist.clear();
    bloblist_joined.clear();

    // Connected component analysis
    // Scan the FG mask to find blob pixels
    for (int x = 0; x < fgmask.rows; x++){
        for (int y = 0; y < fgmask.cols; y++){

            // Extract connected component (blob)
            // We only analyze foreground pixels
            if ((fgmask.at<uchar>(x,y)) == 255.0) {
                pt.x = y;
                pt.y = x;

                // We use the function to obtain the blob.
                floodFill(fgmask, pt, NewValue, &RectangleOutput, MaxMin, MaxMin, Flag);

                // Increse Rectangle size
                int PixelIncrease = 10;
                RectangleOutput.x -= PixelIncrease;
                RectangleOutput.y -= PixelIncrease;
                RectangleOutput.width += PixelIncrease * 2;
                RectangleOutput.height += PixelIncrease * 2;

                // Include blob in 'bloblist'
                bloblist.push_back(RectangleOutput);
            }
        }
    }

    // Iterate through nms until the number of blob do not change
    vector<Rect> resRectsAux1, resRectsAux2;
    resRectsAux1 = bloblist;

    int SizeRectsAux1 = resRectsAux1.size();
    int SizeRectsAux2 = resRectsAux2.size();

    while(SizeRectsAux1 != SizeRectsAux2){
        SizeRectsAux2 = resRectsAux2.size();
        non_max_suppresion(resRectsAux1, resRectsAux2);
        resRectsAux1 = resRectsAux2;
        SizeRectsAux1 = resRectsAux1.size();
    }

    bloblist_joined = resRectsAux2;

    vector<Rect> bloblist_joined_filtered;
    // Suppress small boxes
    for (size_t i = 0; i < bloblist_joined.size(); i++) {
        Rect rect = bloblist_joined[i];
        if (rect.area() > 5000)
            bloblist_joined_filtered.push_back(rect);
    }
    FGBlobs = bloblist_joined_filtered;
    return;
}

void CameraStream::ExtractFGImages(Mat ActualFrame, vector<Rect> FGBlobs){

    FGImages.clear();

    if (FGBlobs.size() == 0){
        EmptyBackground = 1;
        return;
    }

    for (size_t i = 0; i < FGBlobs.size(); i++) {
        Rect r = FGBlobs[i];

        // Check if the new rectangle goes out of the image
        if (r.x < 0)
            r.x = 0;
        if (r.y < 0)
            r.y = 0;
        if ((r.x + r.width) > ActualFrame.cols){
            r.width = ActualFrame.cols - r.x;
        }
        if ((r.y + r.height) > ActualFrame.rows){
            r.height = ActualFrame.rows - r.y;
        }

        Mat NewCamera = ActualFrame(r);
        FGImages.push_back(NewCamera);
    }
    EmptyBackground = 0;
}

void CameraStream::non_max_suppresion(const vector<Rect> &srcRects, vector<Rect> &resRects)
{
    vector<int> IntersectVector (srcRects.size(), 0);
    resRects.clear();

    for (size_t i = 0; i < srcRects.size(); i++) {
        Rect rect1 = srcRects[i];
        bool lonelyBlob = 1;
        for (size_t j = 0; j < srcRects.size(); j++) {
            Rect rect2 = srcRects[j];
            if (i == j){

            }
            else if (((rect1 & rect2).area() > 0) && (IntersectVector[i] == 0) && (IntersectVector[j] == 0)) {
                // They intersect, merge them.
                Rect newrect = rect1 | rect2;
                resRects.push_back(newrect);
                IntersectVector[i] = 1;
                IntersectVector[j] = 1;
                lonelyBlob = 0;
            }
        }
        if(lonelyBlob && IntersectVector[i] == 0)
            resRects.push_back(rect1);
    }
}
