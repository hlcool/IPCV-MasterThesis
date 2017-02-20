clc;
clear all;
close all;

% Parameters
VideoName = 'HallCutted.mpg';

% Videofile Structure
VideoFile = VideoReader(VideoName);

while hasFrame(VideoFile)

    
    Image = readFrame(VideoFile);
    Image = imresize(Image, 2, 'lanczos3');
    Image = imgaussfilt(Image, 1);

    figure(1)
    imshow(Image)
    
    pause();
end