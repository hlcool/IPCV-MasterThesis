clc;
clear all;
close all;

o1 = zeros(986, 1606);
o2 = zeros(986, 1606);
o3 = zeros(986, 1606);

filenames = dir('/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Projected Semantic Frames/Projected Frames 1/*.png');

for i = 1:numel(filenames)
    
    image = imread(['/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Projected Semantic Frames/Projected Frames 1/' filenames(i).name]);
    
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

figure
imshow(uint8(o1))
imwrite(uint8(o1./20),'Sem1Median.png')

filenames = dir('/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Projected Semantic Frames/Projected Frames 2/*.png');

for i = 1:numel(filenames)
    
    image = imread(['/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Projected Semantic Frames/Projected Frames 2/' filenames(i).name]);
    
    md(:,:,i) = image(:,:,1);
end

for r = 1:size(md,1)
    for c = 1:size(md,2)
        vals = md(r,c,:);
        vals = nonzeros(vals(:));
        if (~isempty(vals))
            o2(r,c) = median(vals);
        end
    end
end

figure
imshow(uint8(o2))
imwrite(uint8(o2./20),'Sem2Median.png')

filenames = dir('/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Projected Semantic Frames/Projected Frames 3/*.png');

for i = 1:numel(filenames)
    
    image = imread(['/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Projected Semantic Frames/Projected Frames 3/' filenames(i).name]);
    
    md(:,:,i) = image(:,:,1);
end

for r = 1:size(md,1)
    for c = 1:size(md,2)
        vals = md(r,c,:);
        vals = nonzeros(vals(:));
        if (~isempty(vals))
            o3(r,c) = median(vals);
        end
    end
end

figure
imshow(uint8(o3))
imwrite(uint8(o3./20),'Sem3Median.png')