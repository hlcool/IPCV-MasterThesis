clc;
close all;
figure;
imagesc(Camera10954);
caxis([1 14])
axis off;
figure
imagesc(Camera20007);
caxis([1 14])
axis off;
figure
imagesc(Camera30327);
caxis([1 14])
axis off;




imlegend(hh1, [1,2,3,6,8,14], {'Wall','Floor','Window','Door','Column'})


hh2=imagesc(Camera20007);
axis off;
imlegend(hh2, [1,3,7,14], {'Wall','Floor','Person','Column'})


hh3=imagesc(Camera30327);
axis off;
imlegend(hh3, [1,2,3,8], {'Wall','Building','Floor','Door'})

hh3=imagesc(Camera30327);
axis off;
imlegend(hh3, [1,2,3,6,7,8,14], {'Wall','Building','Floor','Window','Person','Door','Column'})