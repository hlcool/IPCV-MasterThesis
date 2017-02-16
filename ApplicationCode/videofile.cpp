#include "videofile.h"
#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <boost/lexical_cast.hpp>

VideoFile::VideoFile(){}
VideoFile::~VideoFile(){}

using namespace cv;
using namespace std;

void VideoFile::VideoOpenning(string InputPath)
{
    // Open the videofile to check if it exists
    cap.open(InputPath);
    if (!cap.isOpened()) {
        cout << "Could not open video file " << InputPath << endl;
        return;
    }

    // Extract information from VideoCapture
    Width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    Height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    FrameRate = cap.get(CV_CAP_PROP_FPS);
    FrameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);

    // Display information
    cout << "The video to process has the following information:" << endl;
    cout << "Width: " << Width << ". Heigth: " << Height << ". Frames/second: " << FrameRate << endl;
    cout << "The total number of frames is: " << FrameNumber << endl;
}

void VideoFile::decodeBlobFile(string FileName, string FrameNumber)
{
    std::ifstream input(FileName);

    // Auxiliary variables to store the information
    string AuxString;
    int x2, y2;
    double Score;
    Rect RectAux;
    size_t found;
    int Counter = 0;
    int LineCounter = 0;

    // Start decoding the file
    while (input >> AuxString){

        if (AuxString.find("Frame") != std::string::npos)
        {
            // Check if the desired line has been read and so
            // exit the function
            if (LineCounter == atoi(FrameNumber.c_str()))
                return;

            LineCounter++;
        }

        if (LineCounter == atoi(FrameNumber.c_str()))
        {
            //cout << AuxString << endl;
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
                x2 = atoi(AuxString.c_str());
                Counter++;
                break;
            case 4:
                // Case for y2
                found = AuxString.find(']');
                AuxString = AuxString.substr(0, found);
                y2 = atoi(AuxString.c_str());
                Counter++;
                break;
            case 5:
                // Case for "Score:"
                Counter++;
                break;
            case 6:
                // Case for score
                Score = boost::lexical_cast<double>(AuxString);

                // Save blob information into class variables
                RectAux.width = x2 - RectAux.x;
                RectAux.height = y2 - RectAux.y;
                RCNNBoundingBoxes.push_back(RectAux);
                RCNNScores.push_back(Score);

                // Restart the couter to read another blob
                Counter = 1;
                break;

            }
        }
    }
}

void VideoFile::HOGPeopleDetection(Mat ActualFrame)
{
    // initialize the HOG descriptor/person detector
    HOGDescriptor HOG;
    HOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    // Display information about HOG parameters
    if (FlagCOUT == 1){
        cout << "Hog block size: " << HOG.blockSize << endl;
        cout << "Hog cell size: " << HOG.cellSize << endl;
        cout << "Hog number of levels: " << HOG.nlevels << endl;
        cout << "Hog number of bins: " << HOG.nbins << endl;
        cout << "Hog window size: " << HOG.winSize << endl;
        FlagCOUT = 0;
    }

    vector<Rect> BoundingBoxes;
    vector<Rect> BoundingBoxesNMS;

    HOG.detectMultiScale(ActualFrame, BoundingBoxes, 0, Size(8, 8), Size(32, 32), 1.1, 2);
    non_max_suppresion(BoundingBoxes, BoundingBoxesNMS, 0.65);

    //cout << "Numero de BB encontrados: " << BoundingBoxes.size()  << endl;
    for (size_t i = 0; i < BoundingBoxesNMS.size(); i++)
    {
        Rect r = BoundingBoxesNMS[i];
        // The HOG detector returns slightly larger rectangles than the real objects.
        // so we slightly shrink the rectangles to get a nicer output.
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(ActualFrame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 1);
    }
}

void VideoFile::FastRCNNPeopleDetection(Mat ActualFrame, string FrameNumber)
{
    // Decode de txt file for the desired frame number
    string FileName;
    decodeBlobFile(FileName, FrameNumber);

    // Filter blobs by score //

    vector<Rect> BoundingBoxesNMS;
    non_max_suppresion(RCNNBoundingBoxes, BoundingBoxesNMS, 0.65);

    for (size_t i = 0; i < BoundingBoxesNMS.size(); i++)
    {
        Rect r = BoundingBoxesNMS[i];
        rectangle(ActualFrame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 1);
    }


}

void VideoFile::non_max_suppresion(const vector<Rect> &srcRects, vector<Rect> &resRects, float thresh)
{
    resRects.clear();

    const size_t size = srcRects.size();
    if (!size)
        return;

    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.insert(pair<int, size_t>(srcRects[i].br().y, i));
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --end(idxs);
        const Rect& rect1 = srcRects[lastElem->second];

        resRects.push_back(rect1);

        idxs.erase(lastElem);

        for (auto pos = begin(idxs); pos != end(idxs); )
        {
            // grab the current rectangle
            const Rect& rect2 = srcRects[pos->second];

            float intArea = (rect1 & rect2).area();
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {
                pos = idxs.erase(pos);
            }
            else
            {
                ++pos;
            }
        }
    }
}

void VideoFile::maskEnhancement(Mat BackgroundMask)
{
    // Dilatation and Erosion kernels
    Mat kernel_di = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));
    Mat kernel_ero = getStructuringElement(MORPH_ELLIPSE, Size(2, 2), Point(-1, -1));

    // Remove shadows from the mask. Only foreground
    threshold(BackgroundMask, BackgroundMask, 250, 255, THRESH_BINARY);

    // Opening operation
    erode(BackgroundMask, BackgroundMask, kernel_ero, Point(-1, -1));
    dilate(BackgroundMask, BackgroundMask, kernel_di, Point(-1, -1));
}

void VideoFile::imageEnhancement(Mat ActuaFrame)
{
    // Increase video size
    cv::resize(ActualFrame, ActualFrame, {ActualFrame.cols*2, ActualFrame.rows*2}, INTER_LANCZOS4);
    // Remove artifacts by a low-pass filtering
    cv::GaussianBlur(ActualFrame, ActualFrame, Size(1,1), 15);

    Width = ActualFrame.cols;
    Height = ActualFrame.rows;
    if (FlagCOUT == 1){
        cout << "The video has been resize to " << "Width: " << Width << ". Heigth: " << Height << endl;
    }
}

