clc
clear all;
close all;

for Camera = 1 : 3
    
    XML = xml2struct(['Camera ' num2str(Camera) ' File.xml']);
    
    Blobs = XML.DOCANALYSIS.TRAJECTORYSET.TRAJECTORY;
    
    FinalBlobs= cell(1, 1);
    for NBlobs = 1 : size(Blobs, 2)
        for Posiciones = 1 : size(Blobs{1, NBlobs}.POS, 2)
            
            Frame = str2double(Blobs{1, NBlobs}.POS{1, Posiciones}.BOX.FRAME.Text);
            
            Left = Blobs{1, NBlobs}.POS{1, Posiciones}.BOX.LEFT.Text;
            Top = Blobs{1, NBlobs}.POS{1, Posiciones}.BOX.TOP.Text;
            Width = Blobs{1, NBlobs}.POS{1, Posiciones}.BOX.WIDTH.Text;
            Heigth = Blobs{1, NBlobs}.POS{1, Posiciones}.BOX.HEIGHT.Text;
            
            FinalBlobs{Frame + 1, NBlobs} = ['[' Left ', ' Top ', ' Width ', ' Heigth ']'];
        end
    end
    
    fid = fopen(['Camera' num2str(Camera) 'GT.txt'], 'wt' );
    
    for FrameNumber = 1 : size(FinalBlobs, 1)
        fprintf( fid, 'Frame%i ', FrameNumber);
        for j = 1 : size(FinalBlobs, 2)
            
            Blob = FinalBlobs{FrameNumber, j};
            fprintf( fid, '%s ', Blob);
            
        end
        fprintf(fid, '\n');
    end
    
end