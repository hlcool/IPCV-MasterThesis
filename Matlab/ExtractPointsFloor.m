clc;
clear all;
close all;

Image = imread('EmptyCamera3.png');

fig = figure;
imshow(Image)
[xCamera, yCamera] = getpts(fig);
close(fig);
PointsCamera = [xCamera yCamera];
