clc;
clear all;
close all;


%% Pedestrian Detection Multi Camera with semantic constrains

% Camera 1
HOGPrecision1 = [0.152373 0.242678];
HOGRecall1 = [0.164685 0.131695];

DPMPrecision1 = [];
DPMRecall1 = [];

PSPPrecision1 = [];
PSPRecall1 = [];

figure(1)
plot(HOGPrecision1, HOGRecall1, '-rx')
hold on;
plot(DPMPrecision1, DPMRecall1, '-b*')
plot(PSPPrecision1, PSPRecall1, '-gx')
xlim([0 1]);ylim([0 1]);
xlabel('Precision');
ylabel('Recall');
legend('HOG', 'DPM', 'PSP-Net')
title('Camera 1 Pedestrian Detection Multi Camera with semantic')

% Camera 2
HOGPrecision2 = [0.0642081 0.082546];
HOGRecall2 = [0.0872951 0.05533];

DPMPrecision2 = [];
DPMRecall2 = [];

PSPPrecision2 = [];
PSPRecall2 = [];

figure(2)
plot(HOGPrecision2, HOGRecall2, '-rx')
hold on;
plot(DPMPrecision2, DPMRecall2, '-b*')
plot(PSPPrecision2, PSPRecall2, '-gx')
xlim([0 1]);ylim([0 1]);
xlabel('Precision');
ylabel('Recall');
legend('HOG', 'DPM', 'PSP-Net')
title('Camera 2 Pedestrian Detection Multi Camera with semantic')

% Camera 3
HOGPrecision3 = [0.118507 0.187824];
HOGRecall3 = [0.24561 0.19414];

DPMPrecision3 = [];
DPMRecall3 = [];

PSPPrecision3 = [];
PSPRecall3 = [];

figure(3)
plot(HOGPrecision3, HOGRecall3, '-rx')
hold on;
plot(DPMPrecision3, DPMRecall3, '-b*')
plot(PSPPrecision3, PSPRecall3, '-gx')
xlim([0 1]);ylim([0 1]);
xlabel('Precision');
ylabel('Recall');
legend('HOG', 'DPM', 'PSP-Net')
title('Camera 3 Pedestrian Detection Multi Camera with semantic')