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

class PeopleDetector
{
public:
    PeopleDetector();
    ~PeopleDetector();

    // HOG People Detection
    HOGDescriptor HOG;
    void HOGPeopleDetection(CameraStream &Camara);


};

#endif // PEOPLEDETECTOR_H
