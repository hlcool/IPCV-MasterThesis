clc;
clear all;
close all;


%% Pedestrian Detection Multi Camera

% Camera 1
HOGPrecision1 = [0.152373 0.242678 0.297133 0.191489 0.0333333];
HOGRecall1 = [0.164685 0.131695 0.0456792 0.00601042 0.000133565];

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
title('Camera 1 Pedestrian Detection Multi Camera')

% Camera 2
HOGPrecision2 = [0.0642081 0.082546 0.101133 0.117647 0.0967742];
HOGRecall2 = [0.0872951 0.05533 0.018822 0.00454324 0.000486776];

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
title('Camera 2 Pedestrian Detection Multi Camera')

% Camera 3
HOGPrecision3 = [0.118507 0.187824 0.279579 0.402542 0.766667];
HOGRecall3 = [0.24561 0.19414 0.0812739 0.0242038 0.00585987];

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
title('Camera 3 Pedestrian Detection Multi Camera')
