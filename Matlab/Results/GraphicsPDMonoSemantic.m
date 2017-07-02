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

figure1 = figure();
axes1 = axes('Parent',figure1);
plot(HOGPrecision1, HOGRecall1, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(DPMPrecision1, DPMRecall1, '-b*', 'MarkerSize', 10, 'LineWidth', 4)
plot(PSPPrecision1, PSPRecall1, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('HOG', 'DPM', 'PSP-Net')
title('Camera 1 Pedestrian Detection Mono Camera with Semantic')
set(axes1,'FontSize', 40,'FontWeight','bold');

% Camera 2
HOGPrecision2 = [0.223342 0.402685 0.677966 0.756757 1 ] + 0.05;
HOGRecall2 = [0.0863216 0.0584131 0.019471 0.0045432 0.000486776];

DPMPrecision2 = [0.0200525 0.0234217 0.0512351 0.158117 0.834101] + 0.05;
DPMRecall2 = [0.159186 0.186966 0.248346 0.243432 0.0881064];

PSPPrecision2 = [0.576906 0.578195 0.611412 0.967341 0.966667];
PSPRecall2 = [0.209963 0.209963 0.201687 0.100925 0.0141165];

figure2 = figure();
axes2 = axes('Parent',figure2);
plot(HOGPrecision2, HOGRecall2, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(DPMPrecision2, DPMRecall2, '-b*', 'MarkerSize', 10, 'LineWidth', 4)
plot(PSPPrecision2, PSPRecall2, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('HOG', 'DPM', 'PSP-Net')
title('Camera 2 Pedestrian Detection Mono Camera with Semantic')
set(axes2,'FontSize', 40,'FontWeight','bold');

% Camera 3
HOGPrecision3 = [0.327098 0.466068 0.681648 0.848485 1]+ 0.05;
HOGRecall3 = [0.164799 0.118981 0.0463694 0.0142675 0.0033121];

DPMPrecision3 = [0.0233628 0.0251848 0.0646747 0.218275 0.664495]+ 0.05;
DPMRecall3 = [0.275329 0.293673 0.390808 0.456605 0.10387];

PSPPrecision3 = [0.688253 0.688253 0.719038 0.848016 0.865426];
PSPRecall3 = [0.465732 0.465377 0.456721 0.321274 0.0463694];

figure3 = figure();
axes3 = axes('Parent',figure3);
plot(HOGPrecision3, HOGRecall3, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(DPMPrecision3, DPMRecall3, '-b*', 'MarkerSize', 10, 'LineWidth', 4)
plot(PSPPrecision3, PSPRecall3, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('HOG', 'DPM', 'PSP-Net')
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
