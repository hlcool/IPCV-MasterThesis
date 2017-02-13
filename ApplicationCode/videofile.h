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

    int Width, Height, FrameRate, FrameNumber;
    string InputPath;

    VideoCapture cap;
    void VideoOpenning(string InputPath);
    // Mat to store the frame to process
    Mat ActualFrame;
    Mat BackgroundMask;

    // Txt file to extract and save information
    ofstream VideoStatsFile;

    // Mixture Of Gaussians Background Substractor
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();

    void maskEnhancement(Mat BackgroundMask);
    void imageEnhancement(Mat ActuaFrame);
    void HOGPeopleDetection(Mat ActualFrame);
    void non_max_suppresion(const vector<Rect> &srcRects, vector<Rect> &resRects, float thresh);

};

#endif // VIDEOFILE_H
