#ifndef CAMERASTREAM_H
#define CAMERASTREAM_H

#include <QObject>
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
    string GlobalPath;
    string InputPath;
    string VideoPath;
    VideoCapture cap;
    void VideoOpenning(string InputPath);
    int Width, Height, FrameRate, FrameNumber;
    int CameraNumber;

    // Mat to store the frame to process
    Mat ActualFrame;
    Mat ActualSemFrame;
    void getActualSemFrame(string FrameNumber);

    // Enhancement methods
    void maskEnhancement(Mat BackgroundMask);
    void imageEnhancement();

    // Mixture Of Gaussians Background Substractor
    Mat BackgroundMask;
    bool EmptyBackground;
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();

    // Homography and Image Wrapping
    int NViews = 9;
    Mat HomographyBetweenViews;
    vector<Mat> CameraViewsVector;
    vector<Mat> HomographyVector;
    Mat Homography;
    void computeHomography();
    void ViewSelection(vector<Mat> HomographyVector);
    void saveWarpImages(Mat ActualFrame, Mat Homography, String FrameNumber, Mat ImageWarping);

    // Semantic Projection
    vector<Mat> ProjectedFullSemanticVector;
    Mat CommonSemantic12, CommonSemantic23, CommonSemantic13;
    Mat CommonSemanticAllCameras;
    vector<Point2f> ProjectedFloorVector;
    int NumberFloorPoints;
    void ProjectSemanticPoints(Mat &CenitalPlane, Mat &SemanticMask, String FrameNumber);
    void drawSemantic(Mat& CenitalPlane);

    // Induced Plane Homography
    void SemanticCommonPoints();
    void ExtractViewScores();
    void ProjectCommonSemantic();

    // Pedestrian mask, blobs and images
    Mat PedestrianMask;
    vector<Rect> FGBlobs;
    vector<Mat> FGImages;
    void extractPDMask(Mat ActualSemFrame);
    void extractFGBlobs(Mat fgmask, string CBOption);
    void non_max_suppresion(const vector<Rect> &srcRects, vector<Rect> &resRects);
    void ExtractFGImages(Mat ActualFrame, vector<Rect> FGBlobs);

    // AKAZE
    Ptr<AKAZE> akazeDescriptor;
    vector<vector<KeyPoint>> AKAZEKeyPointsVector;
    vector<Mat> AKAZEDescriptorsVector;
    void AkazePointsForViewImages();
    void Akaze(Mat Image1, vector<KeyPoint> kpts1, Mat desc1, Mat Image2, int &NMatches, vector<Point2f> &GoodMatchesPoints1, vector<Point2f> &GoodMatchesPoints2, int CameraView);

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

    // ACF Vectors
    vector<Rect> ACFBoundingBoxes;
    vector<double> ACFScores;
};

#endif // CAMERASTREAM_H
