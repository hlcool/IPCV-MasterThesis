clc;
clear all;
close all;

tic

sempath = '/Volumes/Elena HD 1/Alex/hallEPS_2/feat/';
SavePath = '/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Semantic Images/';
listingSemantic = dir(sempath);
NImages = size(listingSemantic,1);

for imageIndex = 1 : NImages
    
    disp(['Processing Frame ' num2str(imageIndex) '/' num2str(NImages)])
    semname = listingSemantic(imageIndex).name;
    
    pos = strfind(semname, '.');
    semname = semname(1:pos-1);
    
    if listingSemantic(imageIndex).bytes > 100
        load([sempath,semname]);
        
        [FinalScores, FinalLabels] = max(data, [], 3);
        exportName = [SavePath semname '.png'];
        imwrite(uint8(FinalLabels),exportName);
    end
end
toc