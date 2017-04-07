clc;
clear all;
close all;

Image = imread('/Users/alex/Desktop/EmptyCamera2.png');

fig = figure;
imshow(Image)
[xCamera, yCamera] = getpts(fig);
close(fig);
PointsCamera = [xCamera yCamera];
