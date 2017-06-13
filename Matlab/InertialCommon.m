clc;
clear all;
close all;

% Camara 1 y 2
NumHeigth = 7;

CommonImage = imread('/Users/alex/Desktop/CommonSemanticAllCameras.png');
CommonImage = CommonImage/20;

for Heigth = 1 : NumHeigth
    
    figure(1)
    imagesc(CommonImage)
    caxis([0 20])
    title('Common areas between 1 and 2')
    pause()
    
    Inertial1 = imread(['/Users/alex/Desktop/Inertial Planes/Camera 1/Inertial' num2str(Heigth) 'Median.png']);
    Inertial2 = imread(['/Users/alex/Desktop/Inertial Planes/Camera 2/Inertial' num2str(Heigth) 'Median.png']);
    Inertial1 = Inertial1/20;
    Inertial2 = Inertial2/20;
    
    Mask = imdilate(CommonImage > 0,strel('disk',3)) & ~CommonImage;  
    
    for i = 1 : size(Inertial1, 1)
        for j = 1 : size(Inertial1, 2)
            
            Label1 = Inertial1(i,j);
            Label2 = Inertial2(i,j);
            
            if((Label1 == Label2) && (Mask(i,j)> 0))
                CommonImage(i,j) = Label1;
            end
        end
    end   
end


% Camara 1 y 3

CommonImage = imread('/Users/alex/Desktop/CommonSemanticAllCameras.png');
CommonImage = CommonImage/20;

for Heigth = 1 : NumHeigth
    
    figure(2)
    imagesc(CommonImage)
    title('Common areas between 1 and 3')
    caxis([0 20])
    pause()
    
    Inertial1 = imread(['/Users/alex/Desktop/Inertial Planes/Camera 1/Inertial' num2str(Heigth) 'Median.png']);
    Inertial3 = imread(['/Users/alex/Desktop/Inertial Planes/Camera 3/Inertial' num2str(Heigth) 'Median.png']);
    Inertial1 = Inertial1/20;
    Inertial3 = Inertial3/20;
    
    Mask = imdilate(CommonImage > 0,strel('disk',3)) & ~CommonImage; 
    
    for i = 1 : size(Inertial1, 1)
        for j = 1 : size(Inertial1, 2)
            
            Label1 = Inertial1(i,j);
            Label3 = Inertial3(i,j);
            
            if((Label1 == Label3) && (Mask(i,j) > 0))
                CommonImage(i,j) = Label1;
            end
        end
    end   
end


% Camara 2 y 3

CommonImage = imread('/Users/alex/Desktop/CommonSemanticAllCameras.png');
CommonImage = CommonImage/20;

for Heigth = 1 : NumHeigth
    
    figure(3)
    imagesc(CommonImage)
    title('Common areas between 2 and 3')
    caxis([0 20])
    pause()
    
    Inertial2 = imread(['/Users/alex/Desktop/Inertial Planes/Camera 2/Inertial' num2str(Heigth) 'Median.png']);
    Inertial3 = imread(['/Users/alex/Desktop/Inertial Planes/Camera 3/Inertial' num2str(Heigth) 'Median.png']);
    Inertial3 = Inertial3/20;
    Inertial2 = Inertial2/20;
    
    Mask = imdilate(CommonImage > 0,strel('disk',3)) & ~CommonImage; 
    
    for i = 1 : size(Inertial2, 1)
        for j = 1 : size(Inertial2, 2)
            
            Label2 = Inertial2(i,j);
            Label3 = Inertial3(i,j);
            
            if((Label2 == Label3) && (Mask(i,j) > 0))
                CommonImage(i,j) = Label2;
            end
        end
    end   
end

% All the cameras

CommonImage = imread('/Users/alex/Desktop/CommonSemanticAllCameras.png');
CommonImage = CommonImage/20;

for Heigth = 1 : NumHeigth
    
    figure(4)
    imagesc(CommonImage)
    title('Common areas between 1, 2 and 3')
    caxis([0 20])
    pause()
    
    Inertial1 = imread(['/Users/alex/Desktop/Inertial Planes/Camera 1/Inertial' num2str(Heigth) 'Median.png']);
    Inertial2 = imread(['/Users/alex/Desktop/Inertial Planes/Camera 2/Inertial' num2str(Heigth) 'Median.png']);
    Inertial3 = imread(['/Users/alex/Desktop/Inertial Planes/Camera 3/Inertial' num2str(Heigth) 'Median.png']);
    Inertial1 = Inertial1/20;
    Inertial2 = Inertial2/20;
    Inertial3 = Inertial3/20;
    
    for i = 1 : size(Inertial2, 1)
        for j = 1 : size(Inertial2, 2)
            Label1 = Inertial1(i,j);
            Label2 = Inertial2(i,j);
            Label3 = Inertial3(i,j);
            
            if((Label2 == Label3 == Label1) && (CommonImage(i,j)== 0))
                CommonImage(i,j) = Label2;
            end
        end
    end   
end