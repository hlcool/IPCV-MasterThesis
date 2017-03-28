clc;
clear all;
close all;

addpath('..');

CameraNumber = 1;

imagepath = ['/Volumes/Iomega HDD/TFM Videos/Sincronizados/Recording 2/Frame Sequence/Camera ' num2str(2) '/'];
sempath = ['/Volumes/Iomega HDD/TFM Videos/Sincronizados/Recording 2/Semantic/Camera ' num2str(2) '/'];

load('objectName150.mat');

listingImages = dir(imagepath);
listingSemantic = dir(sempath);

NImages = size(listingImages,1);

FinalTemporalLabels = zeros(480,640);

for imageIndex = 1 : NImages
    
    disp(['Processing Frame ' num2str(imageIndex) '/' num2str(NImages)])
    imagename = listingImages(imageIndex).name;
    semname = listingSemantic(imageIndex).name;
    
    if listingImages(imageIndex).bytes > 100
        image = imread([imagepath,imagename]);
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
        %
        [FinalScores,FinalLabels] = max(data, [], 3);
        
        FinalTemporalLabels = FinalTemporalLabels + FinalLabels;
        %         figure
        %         imagesc(FinalLabels)
    end
end

FinalTemporalLabels = FinalTemporalLabels ./ NImages;
figure
imagesc(FinalLabels)