%%
% load semantic map
MAPF1 = imread('/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Semantic Images/Camera 1/Camera10000.png');
MAPF2 = imread('/Users/alex/Desktop/TFM Videos/Sincronizados/Recording 5/Semantic Images/Camera 2/Camera20000.png');
% define colors
colores  = MAPF1';
colores2 = MAPF2';
% load H1 y load H2
load('matframe0cam1.mat');
load('matframe0cam2.mat');
% create points [X,Y] ?
[X,Y] = meshgrid(1:480,1:640);
X=X(:);Y=Y(:);
it=1;
w =  H2./H1;
H(:,:,it) = H1;

w2 =  H22./H21;
Hc2(:,:,it) = H21;
clf
for h = 0.001:0.001:10

PT  = [X';Y';ones(1,numel(X))];
PT1 =  H(:,:,it) *PT;
PT2 =  Hc2(:,:,it) *PT;
clf
scatter3(PT1(1,:),PT1(2,:),PT1(3,:),3,colores(:));title(sprintf('%.2f',it))
hold on, scatter3(PT2(1,:),PT2(2,:),PT2(3,:),3,colores2(:));
pause(0.1);
it=it+1;
H(:,:,it) = w.*H(:,:,it-1);
Hc2(:,:,it) = w2.*Hc2(:,:,it-1);
 

end