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

    ofstream VideoStatsFile;
};

#endif // VIDEOFILE_H
