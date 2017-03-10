// How to extract image warping
Camera1.ImageWarping = Mat::zeros(480, 1280, CV_64F);
warpPerspective(Camera1.ActualFrame2, Camera1.ImageWarping, Camera1.Homography, Camera1.ImageWarping.size());
imshow("Warped Image", Camera1.ImageWarping);


// Homography points for Hall1

pts_src.push_back(Point2f(224, 376));
pts_src.push_back(Point2f(557, 393));
pts_src.push_back(Point2f(334, 225));
pts_src.push_back(Point2f(34, 231));
pts_src.push_back(Point2f(76, 180));
pts_src.push_back(Point2f(55, 261));
pts_src.push_back(Point2f(192, 271));
pts_src.push_back(Point2f(220, 216));

// Threshold Original del modelo DPM person.xml
<ScoreThreshold>-1.4423741965421475</ScoreThreshold>
