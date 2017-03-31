#ifndef CAMERASTREAM_H
#define CAMERASTREAM_H

#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

class CameraStream
{
public:
    CameraStream();
    ~CameraStream();

    // Video Variables
    string InputPath;
    VideoCapture cap;
    void VideoOpenning(string InputPath);
    int Width, Height, FrameRate, FrameNumber;
    int CameraNumber;

    // Mat to store the frame to process
    Mat ActualFrame;

    // Enhancement methods
    void maskEnhancement(Mat BackgroundMask);
    void imageEnhancement();

    // Mixture Of Gaussians Background Substractor
    Mat BackgroundMask;
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();
    vector<Rect> FGBlobs;
    vector<Mat> FGImages;
    bool EmptyBackground;
    void extractFGBlobs(Mat fgmask);
    void non_max_suppresion(const vector<Rect> &srcRects, vector<Rect> &resRects);
    void ExtractFGImages(Mat ActualFrame, vector<Rect> FGBlobs);

    // Homography and Image Wrapping
    Mat Homography;
    Mat SemanticImage;
    void computeHomography();
    void saveWarpImages(Mat ActualFrame, Mat Homography, String FrameNumber);

    // Semantic Projection
    vector<Point2f> ProjectedFloorVector;
    int NumberFloorPoints;
    void ProjectFloorPoints();
    void drawSemantic(Mat& CenitalPlane);

    // Homogrpahy Points Saving
    ofstream PtsDstFile;
    ofstream PtsSrcFile;

    // HOG Vectors
    vector<Rect> HOGBoundingBoxes;
    vector<Rect> HOGBoundingBoxesNMS;
    vector<double> HOGScores;

    // Fast RCNN Vectors
    string FastRCNNMethod;
    vector<Rect> RCNNBoundingBoxes;
    vector<Rect> RCNNBoundingBoxesNMS;
    vector<double> RCNNScores;
    void decodeBlobFile(string FileName, string FrameNumber);
    void FastRCNNPeopleDetection(string FrameNumber, string Method);

    // DPM Vectors
    vector<Rect> DPMBoundingBoxes;
    vector<double> DPMScores;

    // Txt file to extract and save information
    ofstream VideoStatsFile;
};

#endif // CAMERASTREAM_H
