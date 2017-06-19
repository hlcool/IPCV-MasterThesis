clc;
clear all;
close all;


%% Pedestrian Detection Mono Camera

% HOG

% Camera 1
HOGPrecision1 = [0.58341 0.71336 0.884319 0.957447 1];
HOGRecall1 = [0.16909 0.1332 0.0459463 0.0060104 0.00013356];

DPMPrecision1 = [0.0140929 0.0182355 0.0681893 0.633588 0.964497];
DPMRecall1 = [0.241007 0.257361 0.328032 0.265777 0.108855];

figure(1)
plot(HOGPrecision1, HOGRecall1, '-rx')
hold on;
plot(DPMPrecision1, DPMRecall1, '-b*')
xlim([0 1]);ylim([0 1]);
xlabel('Precision');
ylabel('Recall');
legend('HOG','DPM')
title('Camera 1 Pedestrian Detection Mono Camera')

% Camera 2
HOGPrecision2 = [0.18009 0.30972 0.407166 0.459016 0.75];
HOGRecall2 = [0.09540 0.0610 0.0202823 0.0045432 0.00048677];

DPMPrecision2 = [0.0105133 0.0154302 0.0536681 0.158117 0.840299];
DPMRecall2 = [0.205613 0.242754 0.303318 0.243432 0.091351];

figure(2)
plot(HOGPrecision2, HOGRecall2, '-rx')
hold on;
plot(DPMPrecision2, DPMRecall2, '-b*')
xlim([0 1]);ylim([0 1]);
xlabel('Precision');
ylabel('Recall');
legend('HOG','DPM')
title('Camera 2 Pedestrian Detection Mono Camera')

% Camera 3
HOGPrecision3 = [0.35194 0.50976 0.689038 0.742188  0.92];
HOGRecall3 = [0.23911 0.1861 0.0784114 0.0241853 0.00585987];

DPMPrecision3 = [0.0120268 0.0161443 0.0603696 0.218275 0.611303];
DPMRecall3 = [0.38032 0.399751 0.492798 0.456605 0.135032];

figure(3)
plot(HOGPrecision3, HOGRecall3, '-rx')
hold on;
plot(DPMPrecision3, DPMRecall3, '-b*')
xlim([0 1]);ylim([0 1]);
xlabel('Precision');
ylabel('Recall');
legend('HOG','DPM')
title('Camera 3 Pedestrian Detection Mono Camera')
