#include "camerastream.h"
#include <string>
#include <QMutex>
#include <fstream>
#include <QThread>
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

CameraStream::CameraStream(){}
CameraStream::~CameraStream(){}

using namespace cv;
using namespace std;

void CameraStream::VideoOpenning(string InputPath)
{
    // Open the videofile to check if it exists
    cap.open(InputPath);
    if (!cap.isOpened()) {
        cout << "Could not open video file " << InputPath << endl;
        exit(EXIT_FAILURE);
    }

    // Extract information from VideoCapture
    Width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    Height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    FrameRate = cap.get(CV_CAP_PROP_FPS);
    FrameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);

    // Load Semantic images for the camera
    vector<size_t> characterLocations;
    for(size_t i =0; i < InputPath.size(); i++){
        if(InputPath[i] == '/')
            characterLocations.push_back(i);
    }

    size_t Pos = characterLocations[characterLocations.size() - 2];
    VideoPath = InputPath.substr(0, Pos);

    // Save Camera Views into a vector
    for(int i = 1; i <= NViews; i++){
        Mat CameraFrame = imread(VideoPath + "/Homography Images/Camera " + to_string(CameraNumber) + "/View " + to_string(i) + ".jpg");
        CameraViewsVector.push_back(CameraFrame);
    }
}

void CameraStream::getActualSemFrame(string FrameNumber)
{
    int FrameNumber2 = atoi(FrameNumber.c_str()) - 1;
    string SemImagesPath;

    if (FrameNumber2 < 10){
        // Add 000 to the path string
        SemImagesPath = VideoPath + "/Semantic Images/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + "000" + to_string(FrameNumber2) + ".png";
    }
    else if (FrameNumber2 < 100){
        // Add 00 to the path string
        SemImagesPath = VideoPath + "/Semantic Images/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + "00" + to_string(FrameNumber2) + ".png";
    }
    else if (FrameNumber2 < 1000){
        // Add 0 to the path string
        SemImagesPath = VideoPath + "/Semantic Images/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + "0" + to_string(FrameNumber2) + ".png";
    }
    else{
        SemImagesPath = VideoPath + "/Semantic Images/Camera " + to_string(CameraNumber) + "/Camera" + to_string(CameraNumber) + to_string(FrameNumber2) + ".png";
    }
    ActualSemFrame = imread(SemImagesPath);

    // Check for invalid input
    if(! ActualSemFrame.data ){
        cout << "Could not open the actual semantic frame with the following path:" << endl;
        cout << SemImagesPath << endl;
        exit(EXIT_FAILURE);
    }
}

void CameraStream::FastRCNNPeopleDetection(string FrameNumber, string Method)
{
    // Clear vectors
    RCNNBoundingBoxes.clear();
    RCNNBoundingBoxesNMS.clear();
    RCNNScores.clear();

    // Decode de txt file for the desired frame number
    size_t slash = InputPath.find_last_of("/");
    size_t point = InputPath.find_last_of(".");
    string FileName = InputPath.substr(slash + 1, point - slash - 1);

    if (!Method.compare("fast"))
        FileName = "/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/People Detection/" + FileName + "fast.txt";
    else if (!Method.compare("accurate"))
        FileName = "/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/People Detection/" + FileName + "Accurate.txt";

    decodeBlobFile(FileName, FrameNumber);

    // Score average
    double average = accumulate( RCNNScores.begin(), RCNNScores.end(), 0.0) / RCNNScores.size();

    // Filter blobs by average
    for (size_t i = 0; i < RCNNBoundingBoxes.size(); i++) {
        if (RCNNScores[i] <= (average - (average * 0.05)) ) {
            RCNNBoundingBoxes.erase(RCNNBoundingBoxes.begin() + i);
            RCNNScores.erase(RCNNScores.begin() + i);
        }
    }
    RCNNBoundingBoxesNMS = RCNNBoundingBoxes;
}

void CameraStream::decodeBlobFile(string FileName, string FrameNumber)
{
    ifstream input(FileName);

    if (!input) {
        // The file does not exists
        cout << "The file containing the FastRCNN blobs does not exist" << endl;
        exit(EXIT_FAILURE);
    }

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

void CameraStream::maskEnhancement(Mat BackgroundMask)
{
    // Dilatation and Erosion kernels
    Mat kernel_di = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));
    Mat kernel_ero = getStructuringElement(MORPH_ELLIPSE, Size(2, 2), Point(-1, -1));

    // Remove shadows from the mask. Only foreground is saved
    threshold(BackgroundMask, BackgroundMask, 250, 255, THRESH_BINARY);

    // Opening morphological operation
    erode(BackgroundMask, BackgroundMask, kernel_ero, Point(-1, -1));
    dilate(BackgroundMask, BackgroundMask, kernel_di, Point(-1, -1));
}

void CameraStream::imageEnhancement()
{
    // Increase video size
    cv::resize(ActualFrame, ActualFrame, {ActualFrame.cols*2, ActualFrame.rows*2}, INTER_LANCZOS4);
    // Remove interpolation artifacts by a low-pass filtering
    cv::GaussianBlur(ActualFrame, ActualFrame, Size(1,1), 15);

    Width = ActualFrame.cols;
    Height = ActualFrame.rows;
}

void CameraStream::extractPDMask(Mat ActualSemFrame)
{
    Mat SemanticImageGray;

    // Find pedestrian mask (label 7)
    cvtColor(ActualSemFrame, SemanticImageGray , CV_BGR2GRAY);
    compare(SemanticImageGray, 7, PedestrianMask, CMP_EQ);
}

void CameraStream::computeHomography()
{
    string MainPath = "/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Homography Images";

    for (int CameraView = 1; CameraView <= NViews; CameraView++){
        vector<Point2f> pts_src, pts_dst;
        string XCoord, YCoord;

        // CAMERA FRAME POINTS
        string FileName = MainPath + "/Camera " + to_string(CameraNumber) + "/View " + to_string(CameraView) + "_PtsSrcFile.txt";
        ifstream input(FileName);

        if (!input) {
            // The file does not exists
            cout << "Problem with the following path:" << endl;
            cout << FileName << endl;
            cout << "The file that should contain homography points for Camera " + to_string(CameraNumber) + " Frame do not exist" << endl;
            exit(EXIT_FAILURE);
        }

        // Start decoding the file with src points
        while (input >> XCoord){
            input >> YCoord;
            Point2f pt;
            pt.x = atoi(XCoord.c_str());
            pt.y = atoi(YCoord.c_str());
            pts_src.push_back(pt);
        }

        // CENITAL FRAME POINTS
        FileName = MainPath + "/Camera " + to_string(CameraNumber) + "/View " + to_string(CameraView) + "_PtsDstFile.txt";
        ifstream input2(FileName);

        if (!input2) {
            // The file does not exists
            cout << "Problem with the following path:" << endl;
            cout << FileName << endl;
            cout << "The file that should contain homography points for Cenital Frame for camera " + to_string(CameraNumber) + " do not exist" << endl;
            exit(EXIT_FAILURE);
        }

        // Start decoding the file with dst points
        while (input2 >> XCoord){
            input2 >> YCoord;
            Point2f pt;
            pt.x = atoi(XCoord.c_str());
            pt.y = atoi(YCoord.c_str());
            pts_dst.push_back(pt);
        }

        if (pts_dst.size() != pts_src.size()){
            cout << "The number of homography points for Camera " + to_string(CameraNumber) + " is not the same in source and destiny" << endl;
            exit(EXIT_FAILURE);
        }

        // Calculate Homography and store it in the vector
        HomographyVector.push_back(findHomography(pts_src, pts_dst, CV_LMEDS));
    }
}

void CameraStream::HomographySelection(vector<Mat> HomographyVector)
{
    // Compare Actual Frame with all the frames used to extract homographies with AKAZE
    // Extract number of correspondant view to index the homography vectors

    int ViewIndex;
    int NMatches;
    int NMatchesMax = 0;
    vector<Point2f> GoodMatchesPoints1, GoodMatchesPoints2;
    vector<Point2f> GoodMatchesPoints1Def, GoodMatchesPoints2Def;

    for (int CameraView = 0; CameraView < NViews; CameraView++){
        GoodMatchesPoints1.clear();
        GoodMatchesPoints2.clear();

        Mat CameraViewImage = CameraViewsVector[CameraView];
        Mat ViewDescriptor = AKAZEDescriptorsVector[CameraView];
        vector<KeyPoint> ViewKeypoints = AKAZEKeyPointsVector[CameraView];

        Akaze(CameraViewImage, ViewKeypoints, ViewDescriptor, ActualFrame, NMatches, GoodMatchesPoints1, GoodMatchesPoints2, CameraView);

        if (NMatches > NMatchesMax){
            ViewIndex = CameraView;
            NMatchesMax = NMatches;
            GoodMatchesPoints1Def.clear();
            GoodMatchesPoints2Def.clear();
            GoodMatchesPoints1Def = GoodMatchesPoints1;
            GoodMatchesPoints2Def = GoodMatchesPoints2;
        }
    }

    if (GoodMatchesPoints1Def.size() > 4){
        // Number of match points between images when selecting homography is more than 4 so we can compute
        // an homogrpahy

        // Now that we know the nearest view with respect with the ActualFrame we have to
        // interpolate/trasnform the homography so it is more accurate
        // Convert the ActualFrame to the view perspective
        Mat HomographyBetweenViews = findHomography(GoodMatchesPoints2Def, GoodMatchesPoints1Def, CV_LMEDS);

        // Convert ActualSemFrame with the computed homography to be similar to the semantic image from the view
        Mat SemWarping;
        warpPerspective(ActualSemFrame, SemWarping, HomographyBetweenViews, ActualSemFrame.size());
        //String ImageName = "/Users/alex/Desktop/Semantic"+to_string(CameraNumber)+ ".png";
        //imwrite(ImageName, SemWarping);
    }
    Homography = HomographyVector[ViewIndex];
}

void CameraStream::saveWarpImages(Mat ActualFrame, Mat Homography, String FrameNumber)
{
    // Extract image warping
    Mat ImageWarping;
    ImageWarping = Mat::zeros(600, 1500, CV_64F);
    warpPerspective(ActualFrame, ImageWarping, Homography, ImageWarping.size());

    String ImageName = "/Users/alex/IPCV-MasterThesis/ApplicationCode/Wrapped Images/Camera " + to_string(CameraNumber) + "/Frame" + FrameNumber + ".png";
    imwrite(ImageName, ImageWarping);
}

void CameraStream::ProjectFloorPoints(Mat &CenitalPlane)
{
    Mat FloorMask;
    Mat SemanticImageGray;
    vector<Point> FloorPoints;
    vector<Point2f> ProjectedFloor;

    // Find floor mask (label 3) and extract floor coordinates (Point format)
    cvtColor(ActualSemFrame, SemanticImageGray , CV_BGR2GRAY);
    compare(SemanticImageGray, 3, FloorMask, CMP_EQ);
    findNonZero(FloorMask == 255, FloorPoints);

    // Convert from Point to Point2f floor coordinates. Auxiliar vector.
    vector<Point2f> FloorPoints2(FloorPoints.begin(), FloorPoints.end());

    // Apply Homography to vector of Points2f to find the projection of the floor
    perspectiveTransform(FloorPoints2, ProjectedFloor, Homography);

    // Fill the global vector
    ProjectedFloorVector = ProjectedFloor;
    // Extract number of Floor Points
    NumberFloorPoints = static_cast<int>(ProjectedFloorVector.size());

    // Extract projected floor mask
    Mat ProjectedFloorMask = Mat::zeros(CenitalPlane.rows, CenitalPlane.cols, CV_8U);

    for (int i = 0 ; i < NumberFloorPoints ; i++){
        Point punto = ProjectedFloorVector[i];
        if ((punto.y > 0 && punto.y < ProjectedFloorMask.rows) && (punto.x > 0 && punto.x < ProjectedFloorMask.cols)){
            ProjectedFloorMask.at<uchar>(punto.y, punto.x) = 255;
        }
    }

    // Create the mask that will be filled
    Mat FilledFloorMask = Mat::zeros(CenitalPlane.rows, CenitalPlane.cols, CV_8U);

    // Dilatation and Erosion kernels to fill the mask
    Mat kernel_di = getStructuringElement(MORPH_ELLIPSE, Size(9, 9), Point(-1, -1));
    Mat kernel_ero = getStructuringElement(MORPH_ELLIPSE, Size(9, 9), Point(-1, -1));

    // First dilate to fill then erode to keep the original contour
    dilate(ProjectedFloorMask, FilledFloorMask, kernel_di, Point(-1, -1));
    erode(FilledFloorMask, FilledFloorMask, kernel_ero, Point(-1, -1));

    // Erase previous points
    FloorPoints.clear();
    ProjectedFloor.clear();
    ProjectedFloorVector.clear();

    // Find floor mask and extract floor coordinates (Point format)
    findNonZero(FilledFloorMask == 255, FloorPoints);

    // Convert from Point to Point2f floor coordinates. Auxiliar vector.
    vector<Point2f> FloorPoints3(FloorPoints.begin(), FloorPoints.end());

    // Fill the global vector
    ProjectedFloorVector = FloorPoints3;
    // Extract number of Floor Points
    NumberFloorPoints = static_cast<int>(ProjectedFloorVector.size());
}

void CameraStream::drawSemantic(Mat &CenitalPlane)
{
    Mat overlay;
    double alpha = 0.3;
    Vec3b Color;

    // Select color depending on the CameraNumber
    if (CameraNumber == 1){
        Color.val[0] = 0;
        Color.val[1] = 255;
        Color.val[2] = 0;
    }
    if (CameraNumber == 2){
        Color.val[0] = 255;
        Color.val[1] = 0;
        Color.val[2] = 0;
    }
    if (CameraNumber == 3){
        Color.val[0] = 0;
        Color.val[1] = 0;
        Color.val[2] = 255;
    }

    // Copy the cenital image to an overlay layer
    CenitalPlane.copyTo(overlay);

    for (int i = 0 ; i < NumberFloorPoints ; i++){
        Point punto = ProjectedFloorVector[i];
        if ((punto.y > 0 && punto.y < overlay.rows) && (punto.x > 0 && punto.x < overlay.cols)){
            overlay.at<Vec3b>(punto.y, punto.x) = Color;
        }
    }

    // Create the convex poligon from array of Point and add transparency to the final image
    addWeighted(overlay, alpha, CenitalPlane, 1 - alpha, 0, CenitalPlane);
}

void CameraStream::AkazePointsForViewImages()
{
    for (int CameraView = 0; CameraView < NViews; CameraView++){
        Mat ViewImage = CameraViewsVector[CameraView];

        vector<KeyPoint> kpts1;
        Mat desc1;

        // Compute AKAZE points for the selected view image
        akazeDescriptor = AKAZE::create();
        akazeDescriptor->detectAndCompute(ViewImage, noArray(), kpts1, desc1);

        // Save descriptors for the view image
        AKAZEDescriptorsVector.push_back(desc1);
        AKAZEKeyPointsVector.push_back(kpts1);
    }
}

void CameraStream::Akaze(Mat Image1, vector<KeyPoint> kpts1, Mat desc1, Mat Image2, int &NMatches, vector<Point2f> &GoodMatchesPoints1, vector<Point2f> &GoodMatchesPoints2, int CameraView)
{
    vector<KeyPoint> kpts2;
    Mat desc2;

    akazeDescriptor->setNOctaves(2);
    akazeDescriptor->setNOctaveLayers(1);
    akazeDescriptor->detectAndCompute(Image2, noArray(), kpts2, desc2);

    //  ------------------  //
    // BRUTE FORCE MATCHER  //
    //  ------------------  //
    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 5);

    vector<DMatch> good_matches;
    for (size_t i = 0; i < nn_matches.size(); ++i) {
        const float ratio = 0.8; // As in Lowe's paper; can be tuned
        if (nn_matches[i][0].distance < ratio * nn_matches[i][1].distance) {
            good_matches.push_back(nn_matches[i][0]);
        }
    }

    //  ------------------  //
    //  MATCHES COORDINATES //
    //  ------------------  //
    NMatches = good_matches.size();

    // Extract point coordinates from the good matches between ActualFrame and the correspondant ViewFrame
    for (size_t j = 0; j < NMatches; ++j) {
        DMatch Match = good_matches[j];

        int Index1 = Match.queryIdx;
        int Index2 = Match.trainIdx;

        Point2f Point1 = kpts1[Index1].pt;
        Point2f Point2 = kpts2[Index2].pt;

        GoodMatchesPoints1.push_back(Point1);
        GoodMatchesPoints2.push_back(Point2);
    }
    /*
    if(CameraNumber == 1){
        Mat res;
        drawMatches(Image1, kpts1, Image2, kpts2, good_matches, res);
        String ImageName = "/Users/alex/Desktop/image"+to_string(CameraView)+ ".png";
        imwrite(ImageName, res);
    }
    */
}

void CameraStream::extractFGBlobs(Mat fgmask, string CBOption)
{
    // Required variables for connected component analysis
    Point pt;
    Rect RectangleOutput;
    Scalar NewValue = 254;
    Scalar MaxMin = 1;
    int Flag = 8;

    // Clear blob list (to fill with this function)
    vector<Rect> bloblist;
    vector<Rect> bloblist_joined;

    bloblist.clear();
    bloblist_joined.clear();

    // Connected component analysis
    // Scan the FG mask to find blob pixels
    for (int x = 0; x < fgmask.rows; x++){
        for (int y = 0; y < fgmask.cols; y++){

            // Extract connected component (blob)
            // We only analyze foreground pixels
            if ((fgmask.at<uchar>(x,y)) == 255.0) {
                pt.x = y;
                pt.y = x;

                // We use the function to obtain the blob.
                floodFill(fgmask, pt, NewValue, &RectangleOutput, MaxMin, MaxMin, Flag);

                // Increse Rectangle size if method is not Semantic
                if(CBOption.compare("Semantic Detector")){
                    int PixelIncrease = 25;
                    RectangleOutput.x -= PixelIncrease;
                    RectangleOutput.y -= PixelIncrease;
                    RectangleOutput.width += PixelIncrease * 4;
                    RectangleOutput.height += PixelIncrease * 4;
                }

                // Include blob in 'bloblist'
                bloblist.push_back(RectangleOutput);
            }
        }
    }

    // Iterate through nms until the number of blob do not change
    vector<Rect> resRectsAux1, resRectsAux2;
    resRectsAux1 = bloblist;

    int SizeRectsAux1 = resRectsAux1.size();
    int SizeRectsAux2 = resRectsAux2.size();

    while(SizeRectsAux1 != SizeRectsAux2){
        SizeRectsAux2 = resRectsAux2.size();
        non_max_suppresion(resRectsAux1, resRectsAux2);
        resRectsAux1 = resRectsAux2;
        SizeRectsAux1 = resRectsAux1.size();
    }

    bloblist_joined = resRectsAux2;

    vector<Rect> bloblist_joined_filtered;
    // Suppress small boxes
    for (size_t i = 0; i < bloblist_joined.size(); i++) {
        Rect rect = bloblist_joined[i];
        if (rect.area() > 5000)
            bloblist_joined_filtered.push_back(rect);
    }
    FGBlobs = bloblist_joined_filtered;
    return;
}

void CameraStream::ExtractFGImages(Mat ActualFrame, vector<Rect> FGBlobs){

    FGImages.clear();

    if (FGBlobs.size() == 0){
        EmptyBackground = 1;
        return;
    }

    for (size_t i = 0; i < FGBlobs.size(); i++) {
        Rect r = FGBlobs[i];

        // Check if the new rectangle goes out of the image
        if (r.x < 0)
            r.x = 0;
        if (r.y < 0)
            r.y = 0;
        if ((r.x + r.width) > ActualFrame.cols){
            r.width = ActualFrame.cols - r.x;
        }
        if ((r.y + r.height) > ActualFrame.rows){
            r.height = ActualFrame.rows - r.y;
        }

        Mat NewCamera = ActualFrame(r);
        FGImages.push_back(NewCamera);
    }
    EmptyBackground = 0;
}

void CameraStream::non_max_suppresion(const vector<Rect> &srcRects, vector<Rect> &resRects)
{
    // NMS WORKING. Not the one in people detector
    vector<int> IntersectVector (srcRects.size(), 0);
    resRects.clear();

    for (size_t i = 0; i < srcRects.size(); i++) {
        Rect rect1 = srcRects[i];
        bool lonelyBlob = 1;
        for (size_t j = 0; j < srcRects.size(); j++) {
            Rect rect2 = srcRects[j];
            if (i == j){

            }
            else if (((rect1 & rect2).area() > 0) && (IntersectVector[i] == 0) && (IntersectVector[j] == 0)) {
                // They intersect, merge them.
                Rect newrect = rect1 | rect2;
                resRects.push_back(newrect);
                IntersectVector[i] = 1;
                IntersectVector[j] = 1;
                lonelyBlob = 0;
            }
        }
        if(lonelyBlob && IntersectVector[i] == 0)
            resRects.push_back(rect1);
    }
}
