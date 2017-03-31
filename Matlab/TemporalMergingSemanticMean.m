clc;
clear all;
close all;

tic
for CameraNumber = 1 : 3
    % Windows
    %sempath = ['F:\TFM Videos\Sincronizados\Recording 2\Semantic\Camera ' num2str(CameraNumber) '\'];
    
    load('objectName150.mat');
    listingSemantic = dir(sempath);
    
    NImages = size(listingSemantic,1);
    FinalTemporalLabels = zeros(480,640);
    
    Counter = 1;
    for imageIndex = 1 : NImages
        
        disp(['Processing Frame ' num2str(imageIndex) '/' num2str(NImages)])
        semname = listingSemantic(imageIndex).name;
        
        if listingSemantic(imageIndex).bytes > 100
            
            load([sempath,semname]);
            for j = 1 : size(objectNames,1)
                
                FinalTemporalLabels(:,:,j) = FinalTemporalLabels(:,:,j) + data(:,:,j);
                
            end
            
        end
    end
    
    for j = 1 : size(objectNames,1)
        
        FinalTemporalLabels(:,:,j) = FinalTemporalLabels(:,:,j) ./ NImages;
        
    end
    
    FinalTemporalLabels = max(FinalTemporalLabels,3);
    
    exportName = ['Camera ' num2str(CameraNumber) '.png'];
    imwrite(uint8(FinalTemporalLabels),exportName);
    
    figure
    imagesc(FinalTemporalLabels)
    axis off;
    set(gca,'position',[0 0 1 1],'units','normalized')
    title(['Camera ' num2str(CameraNumber)])
    
    clear all;
    clc;
    
end
toc