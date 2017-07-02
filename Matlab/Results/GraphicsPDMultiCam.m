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
title('Camera 1 Pedestrian Detection Multi Camera')
set(axes1,'FontSize', 40,'FontWeight','bold');

% Camera 2
HOGPrecision2 = [0.0642081 0.082546 0.101133 0.117647 0.0967742];
HOGRecall2 = [0.0872951 0.05533 0.018822 0.00454324 0.000486776];

DPMPrecision2 = [0.0026211 0.00392086 0.0147401 0.0696602 0.227843];
DPMRecall2 = [0.209313 0.23465 0.317412 0.240143 0.088431];

PSPPrecision2 = [0.130907 0.140102 0.174297 0.1606 0.0984405 ];
PSPRecall2 = [0.461401 0.461813 0.425328 0.126724 0.0163881];

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
title('Camera 2 Pedestrian Detection Multi Camera')
set(axes2,'FontSize', 40,'FontWeight','bold');

% Camera 3
HOGPrecision3 = [0.118507 0.187824 0.279579 0.402542 0.766667];
HOGRecall3 = [0.24561 0.19414 0.0812739 0.0242038 0.00585987];

DPMPrecision3 = [0.00412866 0.00546056 0.0230953 0.0910185 0.260297];
DPMRecall3 = [0.477038 0.485755 0.554975 0.477525 0.15598];

PSPPrecision3 = [0.111191 0.119067 0.155856 0.321039 0.353623];
PSPRecall3 = [0.585347 0.585139 0.572977 0.388537 0.0932484];

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
