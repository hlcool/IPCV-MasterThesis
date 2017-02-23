clc;
clear all;
close all;


CenitalView = imread('CenitalViewPoints.png');

Image = imread('EmptyHall.png');
Image = imresize(Image, 2, 'lanczos3');
Image = imgaussfilt(Image, 1);

fig = figure;
imshow(Image)
[xCamera, yCamera] = getpts(fig);
close(fig);
PointsCamera = [xCamera yCamera];


% fig2 = figure;
% imshow(CenitalView)
% [xCenital, yCenital] = getpts(fig2);
% close(fig2);
% PointsCenital = [xCenital yCenital];