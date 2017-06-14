#ifndef CAMERAWORKER_H
#define CAMERAWORKER_H

#include <QObject>
#include "barrier.h"
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <fstream>
#include <string>
#include "camerastream.h"
#include "peopledetector.h"
#include "evaluation.h"

using namespace cv;
using namespace std;

class CameraWorker : public QObject
{
    Q_OBJECT
public:
    CameraWorker(CameraStream Camera, Barrier barrier);
    ~CameraWorker();

    // Camera Class
    CameraStream Camera;
    // People detector class
    PeopleDetector PeopleDetec;
    // Evaluation Class
    Evaluation Evaluate;

    // Display Widget Variables
    int WidgetWidth, WidgetHeight;

    // UI Variables
    bool PDFiltering = 0;
    bool SemanticDisplay = 0;
    bool GTDisplay = 0;
    String CBOption;
    // Representation Method
    String RepresentationOption;
    // Pedestrian Filtering
    bool MultiCameraFiltering = 0;
    bool SemanticFiltering = 0;

    // Cenital Frame
    Mat CenitalPlane;
    // Semantic Mask
    Mat SemanticMask;

    // Final Pedestrian Projected Bounding Boxes from the other cameras
    vector<Point2f> ProjCenterPoints1, ProjLeftPoints1, ProjRightPoints1;
    vector<Point2f> ProjCenterPoints2, ProjLeftPoints2, ProjRightPoints2;

    void processVideo();

signals:
    // frame and index of label which frame will be displayed
    void frameFinished(Mat frame, Mat CenitalPlane, int CameraNumber);
    void finished();
    void PedestrianDetectionFinished(int CameraNumber);

public slots:
    void preProcessVideo();


private:
    Barrier barrier;
    // Txt file to extract and save information
    ofstream VideoStatsFile;
};

#endif // CAMERAWORKER_H
