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

void Evaluation::GTTextParser(int CameraNumber, vector<Rect> &GroundTruthVector, String FrameNumber)
{
    //String GTPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Ground-Truth/Camera" + to_string(CameraNumber) + "GT.txt";
    String GTPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Ground-Truth/Camera" + to_string(CameraNumber) + "GT Sin.txt";
    ifstream input(GTPath);

    if (!input) {
        // The file does not exists
        cout << "The file containing the FastRCNN blobs does not exist" << endl;
        exit(EXIT_FAILURE);
    }

    // Auxiliary variables to store the information
    string AuxString;
    Rect RectAux;
    size_t found;
    int Counter = 0;
    int LineCounter = 0;

    // Start decoding the file
    while (input >> AuxString){
        if (AuxString.find("Frame") != std::string::npos) {
            // Check if the desired line has been read and so
            // exit the function
            if (LineCounter == atoi(FrameNumber.c_str()))
                return;
            LineCounter++;
        }
        if (LineCounter == atoi(FrameNumber.c_str())) {
            switch(Counter)
            {
            case 0:
                Counter++;
                break;
            case 1:
                // Case for x1
                found = AuxString.find(',');
                AuxString = AuxString.substr(1, found - 1 );
                RectAux.x = atoi(AuxString.c_str());
                Counter++;
                break;
            case 2:
                // Case for y1
                found = AuxString.find(',');
                AuxString = AuxString.substr(0, found);
                RectAux.y = atoi(AuxString.c_str());
                Counter++;
                break;
            case 3:
                // Case for x2
                found = AuxString.find(',');
                AuxString = AuxString.substr(0, found);
                RectAux.width = atoi(AuxString.c_str());
                Counter++;
                break;
            case 4:
                // Case for y2
                found = AuxString.find(']');
                AuxString = AuxString.substr(0, found);
                RectAux.height = atoi(AuxString.c_str());

                GroundTruthVector.push_back(RectAux);

                // Restart the couter to read another blob
                Counter = 1;
                break;
            }
        }
    }
}

void Evaluation::ExtractEvaluationScores(vector<Rect> GroundTruthVector, vector<Rect> DetectionsVector, String FrameNumber)
{
    float Precision, Recall;
    bool Flag;

    NDetecciones = NDetecciones + DetectionsVector.size();
    GTItems = GTItems + GroundTruthVector.size();

    if(DetectionsVector.empty() && GroundTruthVector.empty()){
        EvaluationFile << FrameNumber << endl;
        return;
    }
    else if(!DetectionsVector.empty() && GroundTruthVector.empty()){
        FalsePositives = FalsePositives + DetectionsVector.size();
    }
    else{
        for (int j = 0; j < GroundTruthVector.size(); j++){
            Rect GT = GroundTruthVector[j];
            Flag = 0;
            for(int i = 0; i < DetectionsVector.size(); i++){
                Rect Detection = DetectionsVector[i];
                if(IoU( GT, Detection, IoUThreshold)){
                    TruePositives++;
                    Flag = 1;
                }
            }
            if(!Flag){
                FalseNegatives++;
            }
        }
    }

    FalsePositives = NDetecciones - TruePositives;

    Precision = TruePositives / (TruePositives + FalsePositives);
    Recall = TruePositives / (TruePositives + FalseNegatives);

    // Save measures to .txt file
    EvaluationFile << FrameNumber << "               " << GTItems << "                    "
                   << TruePositives << "               "
                   << FalsePositives << "                  " << NDetecciones << "                  "
                   << FalseNegatives << "              "
                   << Precision << "                " << Recall << endl;
}

bool Evaluation::IoU(Rect GroundTruth, Rect BoundingBox, float threshold)
{
    // Interseccion entre los dos rectangulos
    float Intersection = (GroundTruth & BoundingBox).area();

    // Suma del area de los dos rectangulos - la itnerseccion
    float Union = GroundTruth.area() + BoundingBox.area() - Intersection;

    // Intersection over Union
    float IoU = Intersection / Union;

    if(IoU > threshold)
        return 1;
    else
        return 0;
}


