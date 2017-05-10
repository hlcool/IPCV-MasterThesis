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

    // Display Widget Variables
    int WidgetWidth, WidgetHeight;

    // UI Variables
    bool PDFiltering;
    String CBOption;
    // Representation Method
    String RepresentationOption;
    String FastRCNNMethod;

    // Cenital Frame
    Mat CenitalPlane;
    // Semantic Mask
    Mat SemanticMask;

    void processVideo();

signals:
    // frame and index of label which frame will be displayed
    void frameFinished(Mat frame, Mat CenitalPlane, int CameraNumber);
    void finished();

public slots:
    void preProcessVideo();


private:
    Barrier barrier;

    // Txt file to extract and save information
    ofstream VideoStatsFile;
};

#endif // CAMERAWORKER_H
