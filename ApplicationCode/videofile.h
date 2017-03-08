#ifndef VIDEOFILE_H
#define VIDEOFILE_H

#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <DPM/dpm.hpp>
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;
using namespace cv::dpm;

class VideoFile
{

public:
    VideoFile();
    ~VideoFile();

    // Video Variables
    string InputPath;
    VideoCapture cap;
    int Width, Height, FrameRate, FrameNumber;
    int CameraNumber;
    void VideoOpenning(string InputPath);

    // Mat to store the frame to process
    Mat ActualFrame;

    // Enhancement methods
    void maskEnhancement(Mat BackgroundMask);
    void imageEnhancement();

    // Mixture Of Gaussians Background Substractor
    Mat BackgroundMask;
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();

    // Homography and Image Wrapping
    Mat CenitalPlane;
    Mat Homography;
    Mat ImageWarping;
    vector<Point2f> ProjectedPoints;
    void computeHomography();
    void projectBlobs(vector<Rect> BoundingBoxes, vector<double> scores, Mat Homography, string Color);
    void projectSemantic();

    // Homogrpahy Points Saving
    ofstream PtsDstFile;
    ofstream PtsSrcFile;

    // Gaussians creation
    Mat GaussianImage;
    void meshgrid(Mat &X, Mat &Y, int rows, int cols);
    void gaussianFunction(Mat &Gaussian3C, Mat X, Mat Y, Point2f center, double score);

    // HOG People Detection
    HOGDescriptor HOG;
    vector<Rect> HOGBoundingBoxes;
    vector<Rect> HOGBoundingBoxesNMS;
    vector<double> HOGScores;
    void HOGPeopleDetection(Mat ActualFrame);

    // Fast RCNN People Detection
    string FastRCNNMethod;
    vector<Rect> RCNNBoundingBoxes;
    vector<Rect> RCNNBoundingBoxesNMS;
    vector<double> RCNNScores;
    void decodeBlobFile(string FileName, string FrameNumber);
    void FastRCNNPeopleDetection(string FrameNumber, string Method);

    // DPM People Detection
    cv::Ptr<DPMDetector> DPMdetector = DPMDetector::create(vector<string>(1, "/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/People Detection/inriaperson.xml"));
    vector<Rect> DPMBoundingBoxes;
    vector<double> DPMScores;
    void DPMPeopleDetection(Mat ActualFrame);

    // Common methods for People Detection
    void paintBoundingBoxes(Mat ActualFrame, string Method, vector<Rect> BoundingBoxes, Scalar Color, int Thickness);
    void non_max_suppresion(const vector<Rect> &srcRects, vector<Rect> &resRects, float thresh);

    // Txt file to extract and save information
    ofstream VideoStatsFile;

    // Flag to cout only once
    int FlagCOUT = 1;
};

#endif // VIDEOFILE_H
