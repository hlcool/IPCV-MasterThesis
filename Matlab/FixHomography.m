clc;
clear all;
close all;

load('MeanCamerasHomography.mat');

% Display the mean images
figure
subplot 311
imshow(uint8(MeanCamera1))
title('Mean Camera 1')
subplot 312
imshow(uint8(MeanCamera2))
title('Mean Camera 2')
subplot 313
imshow(uint8(MeanCamera3))
title('Mean Camera 3')


% Crop images

MeanCamera1Cropped = MeanCamera1(100:500, 20:650, :);
MeanCamera2Cropped = MeanCamera2(100:500, 20:650, :);
MeanCamera3Cropped = MeanCamera3(100:500, 20:650, :);

% Display the mean images
figure
subplot 221
imshow(uint8(MeanCamera1Cropped))
title('Mean Camera 1')
subplot 222
imshow(uint8(MeanCamera2Cropped))
title('Mean Camera 2')
subplot 223
imshow(uint8(MeanCamera3Cropped))
title('Mean Camera 3')


% Select Rombo points
% Camera 1
fig = figure;
imshow(uint8(MeanCamera1Cropped))
[xCamera, yCamera] = getpts(fig);
close(fig);
PointsCamera1 = [xCamera yCamera];

% Camera 2
fig = figure;
imshow(uint8(MeanCamera2Cropped))
[xCamera, yCamera] = getpts(fig);
close(fig);
PointsCamera2 = [xCamera yCamera];

% Camera 3
fig = figure;
imshow(uint8(MeanCamera3Cropped))
[xCamera, yCamera] = getpts(fig);
close(fig);
PointsCamera3 = [xCamera yCamera];

    
