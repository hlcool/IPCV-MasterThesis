clc;
clear all;
close all;


%% Pedestrian Detection Mono Camera

% HOG

% Camera 1
HOGPrecision1 = [0.58341 0.71336 0.884319 0.957447 1];
HOGRecall1 = [0.16909 0.1332 0.0459463 0.0060104 0.00013356];

DPMPrecision1 = [0.0140929];
DPMRecall1 = [0.241007];

figure(1)
plot(1 - HOGPrecision1, HOGRecall1, '-rx')
hold on;
plot(1 - DPMPrecision1, DPMRecall1, '-b*')
xlim([0 1]);ylim([0 1]);
xlabel('1 - Precision');
ylabel('Recall');
legend('HOG','DPM')
title('Camera 1 Pedestrian Detection Mono Camera')

% Camera 2
HOGPrecision2 = [0.18009 0.30972 0.407166 0.459016 0.75];
HOGRecall2 = [0.09540 0.0610 0.0202823 0.0045432 0.00048677];

figure(2)
plot(1 - HOGPrecision2, HOGRecall2, '-rx')
xlim([0 1]);ylim([0 1]);
xlabel('1 - Precision');
ylabel('Recall');
legend('HOG')
title('Camera 2 Pedestrian Detection Mono Camera')

% Camera 3
HOGPrecision3 = [0.35194 0.50976 0.689038 0.742188  0.92];
HOGRecall3 = [0.23911 0.1861 0.0784114 0.0241853 0.00585987];

figure(3)
plot(1 - HOGPrecision3, HOGRecall3, '-rx')
xlim([0 1]);ylim([0 1]);
xlabel('1 - Precision');
ylabel('Recall');
legend('HOG')
title('Camera 3 Pedestrian Detection Mono Camera')
