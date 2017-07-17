#ifndef EVALUATION_H
#define EVALUATION_H

#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

class Evaluation
{
public:
    Evaluation();

    void GTTextParser(int CameraNumber, vector<Rect> &GroundTruthVector, String FrameNumber);

    // Recall and precision vectors
    //vector<double> Recall, Precision;
    float FalsePositives = 0;
    float TruePositives = 0;
    float FalseNegatives = 0;
    float NDetecciones = 0;
    float GTItems = 0;

    double IoUThreshold = 0.3;

    // Extract evaluation scores
    void ExtractEvaluationScores(vector<Rect> GroundTruthVector, vector<Rect> DetectionsVector, String FrameNumber);

    // Intersection Over Union Function
    bool IoU (Rect GroundTruth, Rect BoundingBox, float threshold);

    // Txt file to extract and save information
    ofstream EvaluationFile;
};

#endif // EVALUATION_H
