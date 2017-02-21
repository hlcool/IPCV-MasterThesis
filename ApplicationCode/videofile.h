#ifndef VIDEOFILE_H
#define VIDEOFILE_H

#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

class VideoFile
{

public:
    VideoFile();
    ~VideoFile();

    // Video Variables
    string InputPath;
    VideoCapture cap;
    int Width, Height, FrameRate, FrameNumber;
    void VideoOpenning(string InputPath);

    // Mat to store the frame to process
    Mat ActualFrame;
    Mat ActualFrame2;
    Mat ActualFrameRCNN;

    // Enhancement methods
    void maskEnhancement(Mat BackgroundMask);
    void imageEnhancement(Mat ActuaFrame);

    // Mixture Of Gaussians Background Substractor
    Mat BackgroundMask;
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();

    // Hmography and Image Wrapping
    Mat CenitalPlane;
    Mat Homography;
    Mat ImageWarping;
    vector<Point2f> ProjectedPoints;
    void computeHomography();
    void projectBlobs(vector<Rect> BoundingBoxes, vector<double> scores, Mat Homography);

    // HOG People Detection
    vector<Rect> HOGBoundingBoxes;
    vector<Rect> HOGBoundingBoxesNMS;
    vector<double> HOGScores;
    void HOGPeopleDetection(Mat ActualFrame);

    // Fast RCNN People Detection
    vector<Rect> RCNNBoundingBoxes;
    vector<Rect> RCNNBoundingBoxesNMS;
    vector<double> RCNNScores;
    void decodeBlobFile(string FileName, string FrameNumber);
    void FastRCNNPeopleDetection(string FrameNumber);

    // Common methods for People Detection
    void paintBoundingBoxes(Mat ActualFrame, string Method);
    void non_max_suppresion(const vector<Rect> &srcRects, vector<Rect> &resRects, float thresh);

    // Txt file to extract and save information
    ofstream VideoStatsFile;

    // Flag to cout only once
    int FlagCOUT = 1;

};

#endif // VIDEOFILE_H
