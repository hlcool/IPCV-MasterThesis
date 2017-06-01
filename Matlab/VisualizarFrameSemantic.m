clc;
close all;
clear all;

a = colormap(jet(22));

Path1 = dir('/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Semantic Images/Camera 3/*.png');
Path2 = dir('/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Frame Sequence/Camera 3/*.jpg');

load('objectName150.mat')
for i = 1 : numel(Path1)
    
    Frame = Path2(i).name;
    FrameImage = imread(['/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Frame Sequence/Camera 3/' Frame]);
    
    Semantic = Path1(i).name;
    SemanticImage = imread(['/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 3/Semantic Images/Camera 3/' Semantic]);
    
    figure(1)
    subplot 121
    imshow(FrameImage)
    subplot 122
    imshow(SemanticImage, a)
    pause()
end