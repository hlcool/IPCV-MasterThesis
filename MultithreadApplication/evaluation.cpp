#include "evaluation.h"
#include "camerastream.h"
#include <string>
#include <fstream>
#include <stdio.h>
#include <numeric>
#include <iostream>
#include <stdlib.h>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"

Evaluation::Evaluation(){}

using namespace cv;
using namespace std;

void Evaluation::XMLParser(vector<Rect> &GroundTruthVector)
{

}

bool Evaluation::IoU(Rect GroundTruth, Rect BoundingBox, int threshold)
{
    // Interseccion entre los dos rectangulos
    int Intersection = (GroundTruth & BoundingBox).area();

    // Suma del area de los dos rectangulos - la itnerseccion
    int Union = GroundTruth.area() + BoundingBox.area() - Intersection;

    // Intersection over Union
    int IoU = Intersection / Union;

    if(IoU > threshold)
        return 1;
    else
        return 0;
}


