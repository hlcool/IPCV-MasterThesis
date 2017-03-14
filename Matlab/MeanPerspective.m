clc;
clear all;
close all;


ImagesPath = '/Users/alex/IPCV-MasterThesis/ApplicationCode/Wrapped Images';

% Compute the mean of all the images

% Save all the images into a x,y,3,t matrix

numImages = 300;
for i = 1 : numImages
    % Open image for camera1
    image1 = [ImagesPath '/Camera 1/Frame' num2str(i) '.png'];
    Camera1Frame = imread(image1);
    % Open image for camera2
    image2 = [ImagesPath '/Camera 2/Frame' num2str(i) '.png'];
    Camera2Frame = imread(image2);
    % Open image for camera3
    image3 = [ImagesPath '/Camera 3/Frame' num2str(i) '.png'];
    Camera3Frame = imread(image3);
    
    if i == 1 
        Camara1 = zeros(size(Camera1Frame, 1), size(Camera1Frame, 2), size(Camera1Frame, 3), numImages);
        Camara2 = zeros(size(Camera2Frame, 1), size(Camera2Frame, 2), size(Camera2Frame, 3), numImages);
        Camara3 = zeros(size(Camera3Frame, 1), size(Camera3Frame, 2), size(Camera3Frame, 3), numImages);
    end
    
    Camara1(:, :, :, i) = Camera1Frame;
    Camara2(:, :, :, i) = Camera2Frame;
    Camara3(:, :, :, i) = Camera3Frame;
    
end

% Mean filter for every camera
MeanCamera1 = mean(Camara1,4);
MeanCamera2 = mean(Camara2,4);
MeanCamera3 = mean(Camara3,4);

% Display the mean images
figure
subplot 311
imshow(uint8(MeanCamera1))
title('Mean Camera 1')
subplot 312
imshow(uint8(MeanCamera2))
title('Mean Camera 2')
subplot 313
imshow(uint8(MeanCamera3))
title('Mean Camera 3')
