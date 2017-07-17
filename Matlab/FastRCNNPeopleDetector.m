% Test file to obtain and save the object proposals
clc;
clear all;
close all;

% Parameters
VideoMatrix{1,1} = '/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Videos/Camera1Sync.m2v';
VideoMatrix{1,2} = '/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Videos/Camera2Sync.m2v';
VideoMatrix{1,3} = '/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Videos/Camera3Sync.m2v';
Method = 'fast';

parfor Video = 1 : 3
    
    VideoName = VideoMatrix{1,Video};
    
    % Videofile Structure
    VideoFile = VideoReader(VideoName);
    
    % Open txt file to save the blobs
    fid = fopen( [VideoName(1:strfind(VideoName, '.') - 1) Method '.txt'], 'wt' );
    FrameNumber = 1;
    
    while hasFrame(VideoFile)
        if(FrameNumber > 7360)
            Image = readFrame(VideoFile);
            
            tic
            [ Proposals ] = ProposalExtractor( Image, Method );
            time1 = toc;
            %disp(['Elapsed time for proposal extraction is: ' num2str(time1)])
            
            % FAST_ RCNN
            % [x1, y1, x2, y2]
            tic
            Detections = fast_rcnn_mod( Image, Proposals - 1 );
            time2 = toc;
            %disp(['Elapsed time for people detection is: ' num2str(time2)])
            
            % Export the bounding boxes from Frame to a txt file
            ExportDetections( fid, Detections, FrameNumber );
            
            disp(['Video: ' num2str(Video) '. Computed frame ' num2str(FrameNumber) '/' num2str(round(VideoFile.Duration * VideoFile.FrameRate))...
                ' in ' num2str(time1 + time2) ' seconds.']);
        
        else
            Image = readFrame(VideoFile);
            disp(['Frame: ' num2str(FrameNumber)]);
        end    
        FrameNumber = FrameNumber + 1;
    end
end