% Test file to obtain and save the object proposals
clc;
clear all;
close all;

tic

VideoFile = VideoReader('Hall.avi');

% Open txt file to save the blobs
fid = fopen( 'results.txt', 'wt' );
FrameNumber = 1;
while hasFrame(VideoFile)
    
    Image = readFrame(VideoFile);
    Image = imresize(Image, 2, 'lanczos3');
    Image = imgaussfilt(Image, 1);
    
    [ Proposals ] = ProposalExtractor( Image );
    
    % FAST_ RCNN
    % [x1, y1, x2, y2]
    Detections = fast_rcnn_mod( Image, Proposals - 1 );
    
    % Export the bounding boxes from Frame to a txt file
    ExportDetections( fid, Detections, FrameNumber )
    FrameNumber = FrameNumber + 1;
end
toc

