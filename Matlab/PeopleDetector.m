% Test file to obtain and save the object proposals
clc;
clear all;
close all;

% Parameters
VideoName = 'HallCutted.mpg';
Method = 'accurate';

% Videofile Structure
VideoFile = VideoReader(VideoName);

% Open txt file to save the blobs
fid = fopen( [VideoName(1:strfind(VideoName, '.') - 1) Method '.txt'], 'wt' );
FrameNumber = 1;

while hasFrame(VideoFile)
    tic
    
    Image = readFrame(VideoFile);
    Image = imresize(Image, 2, 'lanczos3');
    Image = imgaussfilt(Image, 1);
    
    [ Proposals ] = ProposalExtractor( Image, Method );
    
    % FAST_ RCNN
    % [x1, y1, x2, y2]
    Detections = fast_rcnn_mod( Image, Proposals - 1 );
    
    % Export the bounding boxes from Frame to a txt file
    ExportDetections( fid, Detections, FrameNumber );
    
    disp(['Computed frame ' num2str(FrameNumber) '/' num2str(round(VideoFile.Duration * VideoFile.FrameRate))...
        ' in ' num2str(toc) ' seconds.']);
    
    FrameNumber = FrameNumber + 1;
end