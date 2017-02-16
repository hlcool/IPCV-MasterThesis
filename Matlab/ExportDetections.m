function [ ] = ExportDetections( fid, Detections, FrameNumber )
% Function to write into a txt file (fid) the detections for a frame

fprintf( fid, 'Frame%i', FrameNumber);
for Blob = 1 : size(Detections, 1)
    x1 = Detections(Blob,1);
    y1 = Detections(Blob,2);
    x2 = Detections(Blob,3);
    y2 = Detections(Blob,4);
    score = Detections(Blob,5);
    fprintf( fid, ' [%.0f, %.0f, %.0f, %.0f] Score: %.4f ', x1, y1, x2, y2, score);
end
fprintf(fid, '\n');


end

