#include "peopledetector.h"
#include "camerastream.h"
#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <boost/lexical_cast.hpp>

PeopleDetector::PeopleDetector(){}
PeopleDetector::~PeopleDetector(){}


void PeopleDetector::HOGPeopleDetection(CameraStream &Camara)
{
    // Clear vectors
    Camara.HOGBoundingBoxes.clear();
    Camara.HOGBoundingBoxesNMS.clear();

    // Initialice the SVM
    HOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    // HOG Detector
    HOG.detectMultiScale(Camara.ActualFrame, Camara.HOGBoundingBoxes, Camara.HOGScores, 0, Size(8, 8), Size(32, 32), 1.1, 2);
    Camara.HOGBoundingBoxesNMS = Camara.HOGBoundingBoxes;
    //non_max_suppresion(HOGBoundingBoxes, HOGBoundingBoxesNMS, 0.65);
}
