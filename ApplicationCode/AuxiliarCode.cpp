// How to extract image warping
Camera1.ImageWarping = Mat::zeros(480, 1280, CV_64F);
warpPerspective(Camera1.ActualFrame2, Camera1.ImageWarping, Camera1.Homography, Camera1.ImageWarping.size());
imshow("Warped Image", Camera1.ImageWarping);

// Algoritmo para sacar un contorno por POlyDP method
vector< vector<Point> > contours;
vector< vector<Point> > app;
// Find contours
findContours( ProjectedFloorMask, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

app.resize(contours.size());
double epsilon = 0.1 * arcLength(Mat(contours[0]), 1);

for( size_t k = 0; k < contours.size(); k++ )
approxPolyDP(Mat(contours[k]), app[k], epsilon, 1);

//drawContours(Newmask, app, -1, Scalar(255), CV_FILLED);
