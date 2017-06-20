clc;
clear all;
%close all;


%% Pedestrian Detection Mono Camera Semantic Constrains

% Camera 1
HOGPrecision1 = [0.500374 0.620092 0.85214 0.944444 1];
HOGRecall1 = [0.0893549 0.0717243 0.0292507 0.0045412 0.000133565];

DPMPrecision1 = [0.028032 0.0294195 0.069287 0.633588 0.992764];
DPMRecall1 = [0.154541 0.162377 0.210086 0.265777 0.0916255];

PSPPrecision1 = [0.52237];
PSPRecall1 = [0.33992];

figure(4)
plot(HOGPrecision1, HOGRecall1, '-rx')
hold on;
plot(DPMPrecision1, DPMRecall1, '-b*')
plot(PSPPrecision1, PSPRecall1, '-gx')
xlim([0 1]);ylim([0 1]);
xlabel('Precision');
ylabel('Recall');
legend('HOG', 'DPM', 'PSP-Net')
title('Camera 1 Pedestrian Detection Mono Camera with semantic constraint')

% Camera 2
HOGPrecision2 = [0.223342 0.402685 0.677966 0.756757 1 ];
HOGRecall2 = [0.0863216 0.0584131 0.019471 0.0045432 0.000486776];

DPMPrecision2 = [0.0200525 0.0234217 0.0512351 0.158117 0.834101];
DPMRecall2 = [0.159186 0.186966 0.248346 0.243432 0.0881064];

PSPPrecision2 = [0.32854];
PSPRecall2 = [0.41148];

figure(5)
plot(HOGPrecision2, HOGRecall2, '-rx')
hold on;
plot(DPMPrecision2, DPMRecall2, '-b*')
plot(PSPPrecision2, PSPRecall2, '-gx')
xlim([0 1]);ylim([0 1]);
xlabel('Precision');
ylabel('Recall');
legend('HOG', 'DPM', 'PSP-Net')
title('Camera 2 Pedestrian Detection Mono Camera with semantic constraint')

% Camera 3
HOGPrecision3 = [0.327098 0.466068 0.681648 0.848485 1];
HOGRecall3 = [0.164799 0.118981 0.0463694 0.0142675 0.0033121];

DPMPrecision3 = [0.0233628 0.0251848 0.0646747 0.218275 0.664495];
DPMRecall3 = [0.275329 0.293673 0.390808 0.456605 0.10387];

PSPPrecision3 = [0.44151];
PSPRecall3 = [0.51502];

figure(6)
plot(HOGPrecision3, HOGRecall3, '-rx')
hold on;
plot(DPMPrecision3, DPMRecall3, '-b*')
plot(PSPPrecision3, PSPRecall3, '-gx')
xlim([0 1]);ylim([0 1]);
xlabel('Precision');
ylabel('Recall');
legend('HOG', 'DPM', 'PSP-Net')
title('Camera 3 Pedestrian Detection Mono Camera with semantic constraint')
