clc;
clear all;
close all;

tic
for CameraNumber = 2 : 3

    %imagepath = ['F:\TFM Videos\Sincronizados\Recording 2\Frame Sequence\Camera ' num2str(CameraNumber) '\'];
    sempath = ['F:\TFM Videos\Sincronizados\Recording 2\Semantic\Camera ' num2str(CameraNumber) '\'];
    
    %load('objectName150.mat');
    
    %listingImages = dir(imagepath);
    listingSemantic = dir(sempath);
    
    NImages = size(listingSemantic,1);
    FinalTemporalLabels = zeros(480,640);
    
    Counter = 1;
    for imageIndex = 1 : NImages
        
        disp(['Processing Frame ' num2str(imageIndex) '/' num2str(NImages)])
        %imagename = listingImages(imageIndex).name;
        semname = listingSemantic(imageIndex).name;

        if listingSemantic(imageIndex).bytes > 100
            %image = imread([imagepath,imagename]);
            load([sempath,semname]);
            
            %         figure
            %         subplot 551
            %         imshow(image)
            %         for j=1:size(data,3)
            %             subplot(5, 5, 1+j)
            %             imagesc(data(:, :, j))
            %             title([objectNames{j, 1} ' = ' num2str(j)])
            %             axis off
            %         end

            [FinalScores,FinalLabels] = max(data, [], 3);
            FinalTemporalLabels (:,:, Counter) = FinalLabels;
            
            if Counter == 400
                tic
                FinalTemporalLabels = mode(FinalTemporalLabels,3);
                toc
                disp('Mode calculated');
                Counter = 2;
            else
                Counter = Counter + 1;
            end
            
            %         figure
            %         imagesc(FinalLabels)
        end
    end
    
    % Coger la moda para un pixel en todos los canales
    FinalTemporalLabels = mode(FinalTemporalLabels,3);
    
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