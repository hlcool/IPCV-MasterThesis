clc;
close all;
clear all;

Path = dir('/Volumes/Elena HD 1/Alex/hallEPS_2/feat/*.mat');

load('objectName150.mat')
for i = 5097 : numel(Path)
    
    ImageName = Path(i).name;
    image = load(['/Volumes/Elena HD 1/Alex/hallEPS_2/feat/' ImageName]);
    image = image.data;
    
    figure(1)
    counter = 1;
    for n = [1,3,14]
        
        subplot(1,3,counter)
        imagesc(image(:,:,n))
        title(objectNames{n,1})
        axis off; 
        axis square;
        counter = counter + 1;
    end
    pause();
end