#ifndef PEOPLEDETECTOR_H
#define PEOPLEDETECTOR_H

#include <QMainWindow>
#include <QObject>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <DPM/dpm.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "camerastream.h"

using namespace std;
using namespace cv;
using namespace cv::dpm;

class PeopleDetector
{
public:
    PeopleDetector();
    ~PeopleDetector();

    // Main People Detection Function
    void MainPeopleDetection(CameraStream &Camera, String CBOption, String RepresentationOption, bool PDFiltering, Mat &CenitalPlane);

    // HOG People Detection
    HOGDescriptor HOG;
    void HOGPeopleDetection(CameraStream &Camera);

    // DPM People Detector
    Ptr<DPMDetector> DPMdetector = DPMDetector::create(vector<string>(1, "/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/People Detection/inriaperson.xml"));
    void DPMPeopleDetection(CameraStream &Camera, bool PDFiltering);
    void paintBoundingBoxes(Mat &ActualFrame, string Method, vector<Rect> BoundingBoxes, int CameraNumber, int Thickness);

    // Non-Maximum-Supression
    void non_max_suppresion(const vector<Rect> &srcRects, vector<Rect> &resRects, float thresh);

    // Gaussians creation
    void projectBlobs(vector<Rect> BoundingBoxes, vector<double> scores, Mat Homography, Mat& CenitalPlane, int CameraNumber, String RepresentationOption);
    void meshgrid(Mat &X, Mat &Y, int rows, int cols);
    void gaussianFunction(Mat &Gaussian3C, Mat X, Mat Y, Point2f center, double score, int CameraNumber);
};

#endif // PEOPLEDETECTOR_H
