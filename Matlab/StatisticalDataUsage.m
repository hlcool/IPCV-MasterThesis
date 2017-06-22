clc;
close all;
clear all

Camera1Blobs = textread('/Users/alex/Desktop/Statistical 2/Projected Blobs 1.txt');
Camera2Blobs = textread('/Users/alex/Desktop/Statistical 2/Projected Blobs 2.txt');
Camera3Blobs = textread('/Users/alex/Desktop/Statistical 2/Projected Blobs 3.txt');

StatisticalMap = imread('/Users/alex/Desktop/StatisticalMap.png');
CenitalMap = imread('/Users/alex/IPCV-MasterThesis/ApplicationCode/Inputs/Homography/CenitalViewMeasured.png');


Step = 70;
VectorRows = Step : Step : (size(StatisticalMap, 1) - Step);
VectorCols = Step : Step : (size(StatisticalMap, 2) - Step);
StadisticsCounter = zeros(size(CenitalMap, 1), size(CenitalMap, 2));

FloorVector = zeros(size(Camera1Blobs,1),1);
DoorVector  = zeros(size(Camera1Blobs,1),1);


for Frame = 1 : size(Camera1Blobs,1)
    
    Blobs1 = Camera1Blobs(Frame, :);
    Blobs1(Blobs1 == 0) = [];
    Blobs2 = Camera2Blobs(Frame, :);
    Blobs2(Blobs2 == 0) = [];
    Blobs3 = Camera3Blobs(Frame, :);
    Blobs3(Blobs3 == 0) = [];
    
    
    % Initialize counters
    ColAnt = 1;
    RowAnt = 1;
    CounterCol = 1;
    CounterRow = 1;
    ContadorZonas = zeros(size(VectorRows,2), size(VectorCols,2));
    
    for Row = VectorRows
        ColAnt = 1;
        CounterCol = 1;
        for Col = VectorCols
            MapSection = CenitalMap(RowAnt : Row, ColAnt : Col, :);
            
            % Camera 1
            for n = 1 : 2 : size(Blobs1,2)
                Blob1Col = Blobs1(n);
                Blob1Row = Blobs1(n + 1);
                if((Blob1Row > RowAnt) && (Blob1Row < Row) && (Blob1Col > ColAnt) && (Blob1Col < Col))
                    ContadorZonas(CounterRow, CounterCol) = ContadorZonas(CounterRow, CounterCol) + 1;
                end
                
            end
            % Camera 2
            for n = 1 : 2 :size(Blobs2,2)
                Blob2Col = Blobs2(n);
                Blob2Row = Blobs2(n + 1);
                if((Blob2Row > RowAnt) && (Blob2Row < Row) && (Blob2Col > ColAnt) && (Blob2Col < Col))
                    ContadorZonas(CounterRow, CounterCol) = ContadorZonas(CounterRow, CounterCol) + 1;
                end
                
            end
            % Camera 3
            for n = 1 : 2 :size(Blobs3,2)
                Blob3Col = Blobs3(n);
                Blob3Row = Blobs3(n + 1);
                if((Blob3Row > RowAnt) && (Blob3Row < Row) && (Blob3Col > ColAnt) && (Blob3Col < Col))
                    ContadorZonas(CounterRow, CounterCol) = ContadorZonas(CounterRow, CounterCol) + 1;
                end
                
            end
            
            CounterCol = CounterCol + 1;
            ColAnt = ColAnt + Step;
        end
        % Counter Updates
        RowAnt = RowAnt + Step;
        CounterRow = CounterRow + 1;
    end
    
    
    [rowfind, colfind] = find(ContadorZonas ~= 0);
    StadisticsCounter(VectorRows(rowfind)-Step+1 : VectorRows(rowfind), VectorCols(colfind) - Step+1 : VectorCols(colfind)) = ...
        StadisticsCounter(VectorRows(rowfind)-Step+1 : VectorRows(rowfind), VectorCols(colfind) - Step+1 : VectorCols(colfind)) + size(colfind,1);
    
    
    %     figure(1)
    %     imagesc(StadisticsCounter)
    %     title(['Frame ' num2str(Frame)])
    %pause(0.001)
end

for Frame = 1 : size(Camera1Blobs,1)
    
    Blobs1 = Camera1Blobs(Frame, :);
    Blobs2 = Camera2Blobs(Frame, :);
    Blobs3 = Camera3Blobs(Frame, :);
    
    for NBlob = 1 : 2 : size(Camera2Blobs)
        
        if(NBlob < size(Camera1Blobs,2))
            Blob1X = round(Camera1Blobs(Frame, NBlob));
            Blob1Y = round(Camera1Blobs(Frame, NBlob + 1));
            if(Blob1X == 0 || Blob1Y == 0)
                Blob1X = 1;
                Blob1Y = 1;
            end
        else
            Blob1X = 1;
            Blob1Y = 1;
        end
        
        % Camara 2
        if(NBlob < size(Camera2Blobs,2))
            Blob2X = round(Camera2Blobs(Frame, NBlob));
            Blob2Y = round(Camera2Blobs(Frame, NBlob + 1));
            if(Blob2X == 0 || Blob2Y == 0)
                Blob2X = 1;
                Blob2Y = 1;
            end
        else
            Blob2X = 1;
            Blob2Y = 1;
        end
        
        % Camara 3
        if(NBlob < size(Camera3Blobs,2))
            Blob3X = round(Camera3Blobs(Frame, NBlob));
            Blob3Y = round(Camera3Blobs(Frame, NBlob + 1));
            if(Blob3X == 0 || Blob3Y == 0)
                Blob3X = 1;
                Blob3Y = 1;
            end
        else
            Blob3X = 1;
            Blob3Y = 1;
        end
        
        % Check semantic
        % Floor
        if((StatisticalMap(Blob1Y, Blob1X) == 3) || (StatisticalMap(Blob2Y, Blob2X) == 3) || (StatisticalMap(Blob3Y, Blob3X) == 3))
            FloorVector(Frame,1) =  FloorVector(Frame,1) + 1;
        end
        
        % Doors
        if((StatisticalMap(Blob1Y, Blob1X) == 8) || (StatisticalMap(Blob2Y, Blob2X) == 8) || (StatisticalMap(Blob3Y, Blob3X) == 8))
            DoorVector(Frame,1) =  DoorVector(Frame,1) + 1;
        end
        
    end
    
end


figure1 = figure();
axes1 = axes('Parent',figure1);
plot(1:size(FloorVector,1), FloorVector, 'LineWidth', 3)
title('Pedestrian on the floor')
ylabel('Number of People');xlabel('Frame')
set(axes1,'FontSize', 40,'FontWeight','bold');


figure2 = figure();
axes2 = axes('Parent',figure2);
plot(1:size(FloorVector,1), DoorVector, 'LineWidth', 3)
title('Pedestrian Using doors');
ylabel('Number of People');xlabel('Frame')
set(axes2,'FontSize', 40,'FontWeight','bold');

StadisticsCounter2 = zeros(size(CenitalMap));
StadisticsCounter2(:,:,1) = StadisticsCounter;
StadisticsCounter2(:,:,2) = StadisticsCounter;

Aux = double(CenitalMap) + StadisticsCounter2.*0.5;

figure
imshow(uint8(Aux))

imwrite(uint8(Aux), 'ProbableAreasPath.png', 'PNG')