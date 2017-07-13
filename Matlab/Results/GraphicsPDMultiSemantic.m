clc;
clear all;
close all;


%% Pedestrian Detection Multi Camera with semantic constrains

% Camera 1
HOGPrecision1 = [0.191415 0.297207 0.386809 0.365591 0.166667];
HOGRecall1 = [0.0881528 0.0710565 0.0289836 0.0045412 0.000133565];

DPMPrecision1 = [0.00863031 0.0104616 0.0294974 0.137491 0.503618];
DPMRecall1 = [0.235513 0.216518 0.206661 0.182024 0.0929611];

PSPPrecision1 = [0.347279 0.349618 0.406318 0.690049 0.736082];
PSPRecall1 = [0.305141 0.304927 0.300053 0.225871 0.0476827];

ACFPrecision1 = [0.0910324 0.481989 0.7 0.975684 1];
ACFRecall1 = [0.340999 0.275139 0.132675 0.0428743 0.0034726];

FastPrecision1 = [ 0.236028 0.276398  0.439694 0.495722  0.599852];
FastRecall1 = [ 0.318337 0.312751 0.275527 0.245584 0.216155];

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
title('Camera 1 Pedestrian Detection Multi Camera with Semantic')
set(axes1,'FontSize', 40,'FontWeight','bold');


% Camera 2
HOGPrecision2 = [0.0956998 0.138434 0.191564 0.266667 0.2];
HOGRecall2 = [0.0736654 0.050787 0.0176862 0.00454324 0.000486776];

DPMPrecision2 = [0.00392912 0.00566758 0.0162191 0.0819659 0.307794];
DPMRecall2 = [0.149629 0.162558 0.168579 0.163719 0.0756125];

PSPPrecision2 = [0.229278 0.230057 0.249162 0.251698 0.256637];
PSPRecall2 = [0.201427  0.201557 0.192963 0.0962194 0.0141165];

ACFPrecision2 = [0.054747 0.242336 0.256804 0.190909 0.190909];
ACFRecall2 = [0.334066 0.189198 0.0597112 0.00681486 0];

FastPrecision2 = [ 0.146709 0.20933 0.256674  0.286144 0.285026 ];
FastRecall2 = [ 0.251698 0.316265 0.210612 0.200115 0.124777 ];

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
title('Camera 2 Pedestrian Detection Multi Camera with Semantic')
set(axes2,'FontSize', 40,'FontWeight','bold');

% Camera 3
HOGPrecision3 = [0.141816 0.222598 0.382766 0.615385 0.866667];
HOGRecall3 = [0.153474 0.117452 0.0486624 0.0142675 0.0033121];

DPMPrecision3 = [0.00642076 0.00819795 0.0245744 0.114951 0.313413];
DPMRecall3 = [0.375296 0.37978 0.410244 0.362879 0.120132];

PSPPrecision3 = [0.288972 0.290277 0.324532 0.443826 0.417254];
PSPRecall3 = [0.469624 0.469506 0.458996 0.323694 0.0603822];

ACFPrecision3 = [0.0648929 0.329448 0.536032 0.515419 1];
ACFRecall3 = [0.627692 0.417837 0.168577 0.0298089 0.00254777];

FastPrecision3 = [0.162635 0.222016 0.26613 0.312217 0.360444];
FastRecall3 = [0.524904 0.526779 0.43825 0.407955 0.30382];

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
title('Camera 3 Pedestrian Detection Multi Camera with Semantic')
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