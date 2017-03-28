clc;
clear all;
close all;

addpath('..');

imagepath = '/Volumes/Iomega HDD/TFM Videos/Sincronizados/Recording 2/Frame Sequence/Camera 2/';
imagename = 'Camera24555.jpg';

sempath = '/Volumes/Iomega HDD/TFM Videos/hallEPS/feat/';
semname = 'Camera24555.mat';

image = imread([imagepath,imagename]);
load([sempath,semname]);
unary = -data;

%D = Densecrf(image,unary);

% Some settings.
% D.gaussian_x_stddev = 3;
% D.gaussian_y_stddev = 3;
% D.gaussian_weight = 1; 
% 
% D.bilateral_x_stddev = 60;
% D.bilateral_y_stddev = 60;
% D.bilateral_r_stddev = 10;
% D.bilateral_g_stddev = 10;
% D.bilateral_b_stddev = 10;
% D.bilateral_weight = 1; 
% 
% % Threhold
% figure(1);
% D.threshold;
% D.display();

% Meanfield (faster but less accurate)
% figure(2);
% D.mean_field;
% D.display();

% filtering
%kk = D.segmentation;
dataR = data;
%for j=1:size(data,3),dataR(:,:,j) = data(:,:,j) .*double(kk==j);end

% show results
figure(1)
subplot(5,5,1),imshow(image),for j=1:size(unary,3),subplot(5,5,1+j),imagesc(data(:,:,j));axis off;end
figure(2)
subplot(5,5,1),imshow(image),for j=1:size(unary,3),subplot(5,5,1+j),imagesc(dataR(:,:,j));axis off;end