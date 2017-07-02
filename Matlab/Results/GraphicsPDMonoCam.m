clc;
clear all;
close all;


%% Pedestrian Detection Mono Camera

% Camera 1
HOGPrecision1 = [0.58341 0.71336 0.884319 0.957447 1];
HOGRecall1 = [0.16909 0.1332 0.0459463 0.0060104 0.00013356];

DPMPrecision1 = [0.0140929 0.0182355 0.0681893 0.633588 0.964497];
DPMRecall1 = [0.241007 0.257361 0.328032 0.265777 0.108855];

PSPPrecision1 = [0.378523 0.413583 0.626809 1 1 ];
PSPRecall1 = [0.435956 0.435956 0.422198 0.270736 0.0519567];

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
title('Camera 1 Pedestrian Detection Mono Camera')
set(axes1,'FontSize', 40,'FontWeight','bold');

% Camera 2
HOGPrecision2 = [0.18009 0.30972 0.407166 0.459016 0.75];
HOGRecall2 = [0.09540 0.0610 0.0202823 0.0045432 0.00048677];

DPMPrecision2 = [0.0105133 0.0154302 0.0536681 0.158117 0.840299];
DPMRecall2 = [0.205613 0.242754 0.303318 0.243432 0.091351];

PSPPrecision2 = [0.332649 0.364877 0.45742 0.842213 0.95283];
PSPRecall2 = [0.472984 0.472984 0.436638 0.133377 0.0163881];

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
title('Camera 2 Pedestrian Detection Mono Camera')
set(axes2,'FontSize', 40,'FontWeight','bold');

% Camera 3
HOGPrecision3 = [0.35194 0.50976 0.689038 0.742188  0.92];
HOGRecall3 = [0.23911 0.1861 0.0784114 0.0241853 0.00585987];

DPMPrecision3 = [0.0120268 0.0161443 0.0603696 0.218275 0.611303];
DPMRecall3 = [0.38032 0.399751 0.492798 0.456605 0.135032];

PSPPrecision3 = [0.430561 0.43056 0.490085 0.702815 0.841523];
PSPRecall3 = [0.582485 0.582485  0.572556 0.388025 0.0796843];

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
title('Camera 3 Pedestrian Detection Mono Camera')
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
