clc;
clear all;
close all

NumCameras = 3;
NumHeigth = 7;

for Camera = 1: NumCameras
    for Heigth = 1 : NumHeigth
        
        disp(['Processing Camera ' num2str(Camera) ' with heigth ' num2str(Heigth) '/' num2str(NumHeigth)]);
        
        o1 = zeros(986, 1606);
        
        filenames = dir(['/Users/alex/Desktop/Inertial Planes/Camera ' num2str(Camera) '/Height ' num2str(Heigth) '/*.png']);
        
        for i = 1:numel(filenames)
            
            image = imread(['/Users/alex/Desktop/Inertial Planes/Camera ' num2str(Camera) '/Height ' num2str(Heigth) '/' filenames(i).name]);
            
            md(:,:,i) = image(:,:,1);
        end
        
        for r = 1:size(md,1)
            for c = 1:size(md,2)
                vals = md(r,c,:);
                vals = nonzeros(vals(:));
                if (~isempty(vals))
                    o1(r,c) = median(vals);
                end
            end
        end

        imwrite(uint8(o1),['/Users/alex/Desktop/Inertial Planes/Camera ' num2str(Camera) '/Inertial' num2str(Heigth) 'Median.png'])
    end
end