clc;
clear all;
close all;


%% Pedestrian Detection Mono Camera Semantic Constrains

% Camera 1
PSPPrecisionRaw1 = [0.378595 0.413658 0.626883 1 1];
PSPRecallRaw1 = [0.567629 0.567629 0.549722 0.354659 0.0676287];

PSPPrecisionSemantic1 = [0.576124 0.578766 0.708144 1 1];
PSPRecallSemantic1 = [0.438804 0.438804 0.432371 0.314152 0.0605007];

figure1 = figure();
axes1 = axes('Parent',figure1);
plot(PSPPrecisionRaw1, PSPRecallRaw1, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(PSPPrecisionSemantic1, PSPRecallSemantic1, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('RAW', 'Semantic')
title('Camera 1 Pedestrian Detection Mono Camera')
title('Camera 1 Pedestrian Detection Mono Camera with Semantic')
set(axes1,'FontSize', 40,'FontWeight','bold');

% Camera 2
PSPPrecisionRaw2 = [0.200251 0.21965 0.28688 0.789796 0.924528];
PSPRecallRaw2 = [0.57959 0.57959 0.557464 0.255614 0.0323646];

PSPPrecisionSemantic2 = [0.578323 0.57955 0.613626 0.96997 0.967391];
PSPRecallSemantic2 = [ 0.451123  0.451123 0.43428 0.213342 0.0293923];

figure1 = figure();
axes1 = axes('Parent',figure1);
plot(PSPPrecisionRaw2, PSPRecallRaw2, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(PSPPrecisionSemantic2, PSPRecallSemantic2, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('RAW', 'Semantic')
title('Camera 1 Pedestrian Detection Mono Camera')
title('Camera 1 Pedestrian Detection Mono Camera with Semantic')
set(axes1,'FontSize', 40,'FontWeight','bold');

% Camera 3
PSPPrecisionRaw3 = [ 0.397178 0.397178 0.452723 0.684186 0.541523];
PSPRecallRaw3 = [0.704136 0.704136 0.693129 0.494997 0.104403];


PSPPrecisionSemantic3 = [0.686792 0.686792 0.717671 0.849528 0.708171];
PSPRecallSemantic3 = [0.607071 0.607071 0.596064 0.419947 0.0607071];

figure1 = figure();
axes1 = axes('Parent',figure1);
plot(PSPPrecisionRaw3, PSPRecallRaw3, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(PSPPrecisionSemantic3, PSPRecallSemantic3, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('RAW', 'Semantic')
title('Camera 1 Pedestrian Detection Mono Camera')
title('Camera 1 Pedestrian Detection Mono Camera with Semantic')
set(axes1,'FontSize', 40,'FontWeight','bold');


% F-Scores
FScorePSPRaw1 = max((2 .* PSPPrecisionRaw1 .* PSPRecallRaw1)./(PSPPrecisionRaw1 + PSPRecallRaw1))
FScorePSPSemantic1 = max((2 .* PSPPrecisionSemantic1 .* PSPRecallSemantic1)./(PSPPrecisionSemantic1 + PSPRecallSemantic1))

FScorePSPRaw2 = max((2 .* PSPPrecisionRaw2 .* PSPRecallRaw2)./(PSPPrecisionRaw2 + PSPRecallRaw2))
FScorePSPSemantic2 = max((2 .* PSPPrecisionSemantic2 .* PSPRecallSemantic2)./(PSPPrecisionSemantic2 + PSPRecallSemantic2))

FScorePSPRaw3 = max((2 .* PSPPrecisionRaw3 .* PSPRecallRaw3)./(PSPPrecisionRaw3 + PSPRecallRaw3))
FScorePSPSemantic3 = max((2 .* PSPPrecisionSemantic3 .* PSPRecallSemantic3)./(PSPPrecisionSemantic3 + PSPRecallSemantic3))

