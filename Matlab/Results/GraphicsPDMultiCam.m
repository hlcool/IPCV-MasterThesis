clc;
clear all;
close all;


%% Pedestrian Detection Multi Camera

% Camera 1
HOGPrecision1 = [0.152373 0.242678 0.297133 0.191489 0.0333333];
HOGRecall1 = [0.164685 0.131695 0.0456792 0.00601042 0.000133565];

DPMPrecision1 = [0.00573624 0.00700831 0.0258927 0.0970311 0.349398];
DPMRecall1 = [0.330375 0.313797 0.340906 0.255644 0.108455];

PSPPrecision1 = [0.148363 0.160295 0.211219 0.412793 0.37971];
PSPRecall1 = [0.393939 0.395083 0.384708 0.258409 0.052491];

ACFPrecision1 = [0.057837 0.277369 0.559127 0.781155 0.83871];
ACFRecall1 = [0.500124 0.385411 0.195008 0.0686523 0.00694537];

FastPrecision1 = [ 0.0591427 0.0827778 0.14149 0.2121 0.348687];
FastRecall1 = [ 0.551139 0.506714 0.449668 0.399973 0.338404];

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
title('Camera 1 Pedestrian Detection Multi Camera')
set(axes1,'FontSize', 40,'FontWeight','bold');

% Camera 2
HOGPrecision2 = [0.0642081 0.082546 0.101133 0.117647 0.0967742];
HOGRecall2 = [0.0872951 0.05533 0.018822 0.00454324 0.000486776];

DPMPrecision2 = [0.0026211 0.00392086 0.0147401 0.0696602 0.227843];
DPMRecall2 = [0.209313 0.23465 0.317412 0.240143 0.088431];

PSPPrecision2 = [0.130907 0.140102 0.174297 0.1606 0.0984405 ];
PSPRecall2 = [0.461401 0.461813 0.425328 0.126724 0.0163881];

ACFPrecision2 = [0.0389929 0.146597 0.155325  0.0652504 0];
ACFRecall2 = [0.473175 0.262698 0.0681486 0.00697712 0];

FastPrecision2 = [ 0.0539248 0.0735382 0.121462 0.168227  0.22856                    ];
FastRecall2 = [ 0.612933 0.560749 0.495651 0.410668 0.284115];

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
title('Camera 2 Pedestrian Detection Multi Camera')
set(axes2,'FontSize', 40,'FontWeight','bold');

% Camera 3
HOGPrecision3 = [0.118507 0.187824 0.279579 0.402542 0.766667];
HOGRecall3 = [0.24561 0.19414 0.0812739 0.0242038 0.00585987];

DPMPrecision3 = [0.00412866 0.00546056 0.0230953 0.0910185 0.260297];
DPMRecall3 = [0.477038 0.485755 0.554975 0.477525 0.15598];

PSPPrecision3 = [0.111191 0.119067 0.155856 0.321039 0.353623];
PSPRecall3 = [0.585347 0.585139 0.572977 0.388537 0.0932484];

ACFPrecision3 = [0.0452859 0.20237 0.32344 0.20362 0.15625];
ACFRecall3 = [0.762075 0.535129 0.216285 0.0343949 0.00254777];

FastPrecision3 = [0.0401092 0.055973 0.0945656 0.139574 0.214229];
FastRecall3 = [0.705578 0.65459 0.59481 0.524157 0.410678];

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
title('Camera 3 Pedestrian Detection Multi Camera')
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
