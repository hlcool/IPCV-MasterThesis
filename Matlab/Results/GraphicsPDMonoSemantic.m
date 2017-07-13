clc;
clear all;
close all;


%% Pedestrian Detection Mono Camera Semantic Constrains

% Camera 1
HOGPrecision1 = [0.500374 0.620092 0.85214 0.944444 1] + 0.05;
HOGRecall1 = [0.0893549 0.0717243 0.0292507 0.0045412 0.000133565];

DPMPrecision1 = [0.028032 0.0294195 0.069287 0.633588 0.992764] + 0.05;
DPMRecall1 = [0.154541 0.162377 0.210086 0.265777 0.0916255];

PSPPrecision1 = [0.578294 0.5811 0.708727 1 1];
PSPRecall1 = [0.335916 0.335916 0.33084 0.240283 0.0470148];

ACFPrecision1 = [0.265642 0.978162 1 1 1];
ACFRecall1 = [0.358635  0.28682 0.137669  0.043275 0.00347269];

FastPrecision1 = [ 0.431117 0.598995 0.810931 0.900604 0.997565];
FastRecall1 = [ 0.33382 0.31774 0.293269 0.258982 0.218913];

figure1 = figure();
axes1 = axes('Parent',figure1);
plot(HOGPrecision1, HOGRecall1, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(DPMPrecision1, DPMRecall1, '-b*', 'MarkerSize', 10, 'LineWidth', 4)
plot(PSPPrecision1, PSPRecall1, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
plot(ACFPrecision1, ACFRecall1, '-kx', 'MarkerSize', 10, 'LineWidth', 4)
plot(FastPrecision1, FastRecall1, '-yx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('HOG', 'DPM', 'PSP-Net', 'ACF', 'Fast-RCNN')
title('Camera 1 Pedestrian Detection Mono Camera')
title('Camera 1 Pedestrian Detection Mono Camera with Semantic')
set(axes1,'FontSize', 40,'FontWeight','bold');

% Camera 2
HOGPrecision2 = [0.223342 0.402685 0.677966 0.756757 1 ] + 0.05;
HOGRecall2 = [0.0863216 0.0584131 0.019471 0.0045432 0.000486776];

DPMPrecision2 = [0.0200525 0.0234217 0.0512351 0.158117 0.834101] + 0.05;
DPMRecall2 = [0.159186 0.186966 0.248346 0.243432 0.0881064];

PSPPrecision2 = [0.576906 0.578195 0.611412 0.967341 0.966667];
PSPRecall2 = [0.209963 0.209963 0.201687 0.100925 0.0141165];

ACFPrecision2 = [0.126786 0.737066 1 1 1];
ACFRecall2 = [0.341467 0.195955 0.0687977 0.00827519 0.0001];

FastPrecision2 = [0.507956 0.656303 0.834316 0.914197 0.978286];
FastRecall2 = [0.268652 0.253284 0.229596 0.190167 0.138893];

figure2 = figure();
axes2 = axes('Parent',figure2);
plot(HOGPrecision2, HOGRecall2, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(DPMPrecision2, DPMRecall2, '-b*', 'MarkerSize', 10, 'LineWidth', 4)
plot(PSPPrecision2, PSPRecall2, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
plot(ACFPrecision2, ACFRecall2, '-kx', 'MarkerSize', 10, 'LineWidth', 4)
plot(FastPrecision2, FastRecall2, '-yx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('HOG', 'DPM', 'PSP-Net', 'ACF', 'Fast-RCNN')
title('Camera 2 Pedestrian Detection Mono Camera with Semantic')
set(axes2,'FontSize', 40,'FontWeight','bold');

% Camera 3
HOGPrecision3 = [0.327098 0.466068 0.681648 0.848485 1]+ 0.05;
HOGRecall3 = [0.164799 0.118981 0.0463694 0.0142675 0.0033121];

DPMPrecision3 = [0.0233628 0.0251848 0.0646747 0.218275 0.664495]+ 0.05;
DPMRecall3 = [0.275329 0.293673 0.390808 0.456605 0.10387];

PSPPrecision3 = [0.688253 0.688253 0.719038 0.848016 0.865426];
PSPRecall3 = [0.465732 0.465377 0.456721 0.321274 0.0463694];

ACFPrecision3 = [0.166579 0.886299 0.971761 0.973214 1];
ACFRecall3 = [0.598248 0.395585 0.149045 0.0277495 0.00254777  ];

FastPrecision3 = [0.467148 0.619762 0.779859 0.854696 0.900553];
FastRecall3 = [0.490275 0.462027 0.423988 0.368662 0.290701];

figure3 = figure();
axes3 = axes('Parent',figure3);
plot(HOGPrecision3, HOGRecall3, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(DPMPrecision3, DPMRecall3, '-b*', 'MarkerSize', 10, 'LineWidth', 4)
plot(PSPPrecision3, PSPRecall3, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
plot(ACFPrecision3, ACFRecall3, '-kx', 'MarkerSize', 10, 'LineWidth', 4)
plot(FastPrecision3, FastRecall3, '-yx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('HOG', 'DPM', 'PSP-Net', 'ACF', 'Fast-RCNN')
title('Camera 3 Pedestrian Detection Mono Camera with semantic')
set(axes3,'FontSize', 40,'FontWeight','bold');

% F-Scores
FScoreHOG1 = max((2 .* HOGPrecision1 .* HOGRecall1)./(HOGPrecision1 + HOGRecall1))
FScoreHOG2 = max((2 .* HOGPrecision2 .* HOGRecall2)./(HOGPrecision2 + HOGRecall2))
FScoreHOG3 = max((2 .* HOGPrecision3 .* HOGRecall3)./(HOGPrecision3 + HOGRecall3))

FScoreDPM1 = max((2 .* DPMPrecision1 .* DPMRecall1)./(DPMPrecision1 + DPMRecall1))
FScoreDPM2 = max((2 .* DPMPrecision2 .* DPMRecall2)./(DPMPrecision2 + DPMRecall2))
FScoreDPM3 = max((2 .* DPMPrecision3 .* DPMRecall3)./(DPMPrecision3 + DPMRecall3))

FScorePSP1 = max((2 .* PSPPrecision1 .* PSPRecall1)./(PSPPrecision1 + PSPRecall1))
FScorePSP2 = max((2 .* PSPPrecision2 .* PSPRecall2)./(PSPPrecision2 + PSPRecall2))
FScorePSP3 = max((2 .* PSPPrecision3 .* PSPRecall3)./(PSPPrecision3 + PSPRecall3))

FScoreACF1 = max((2 .* ACFPrecision1 .* ACFRecall1)./(ACFPrecision1 + ACFRecall1))
FScoreACF2 = max((2 .* ACFPrecision2 .* ACFRecall2)./(ACFPrecision2 + ACFRecall2))
FScoreACF3 = max((2 .* ACFPrecision3 .* ACFRecall3)./(ACFPrecision3 + ACFRecall3))

FScoreFast1 = max((2 .* FastPrecision1 .* FastRecall1)./(FastPrecision1 + FastRecall1))
FScoreFast2 = max((2 .* FastPrecision2 .* FastRecall2)./(FastPrecision2 + FastRecall2))
FScoreFast3 = max((2 .* FastPrecision3 .* FastRecall3)./(FastPrecision3 + FastRecall3))
