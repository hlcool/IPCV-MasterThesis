clear all;
close all;

o1 = uint8(zeros(986, 1606));
o2 = uint8(zeros(986, 1606));

filenames = dir('/Users/alex/Desktop/Vistas Proyectadas/*.jpg');

for i = 1:numel(filenames)
    
    image = imread(['/Users/alex/Desktop/Vistas Proyectadas/' filenames(i).name]);
    
    o2 = o2 + image.*0.5;
    md(:,:,i) = image(:,:,1);
end

figure
imshow(o2)


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