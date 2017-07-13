clc;
clear all;
close all;


%% Pedestrian Detection Mono Camera Semantic Constrains

% Camera 1
FastPrecisionRaw1 = [0.174889 0.256889 0.386838 0.482826 0.710518];
FastRecallRaw1 = [0.595246 0.571946 0.527147 0.468984 0.393223];

FastPrecisionSemantic1 = [0.576124 0.578766 0.708144 1 1];
FastRecallSemantic1 = [0.438804 0.438804 0.432371 0.314152 0.0605007];

figure1 = figure();
axes1 = axes('Parent',figure1);
plot(FastPrecisionRaw1, FastRecallRaw1, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(FastPrecisionSemantic1, FastRecallSemantic1, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('RAW', 'Semantic')
title('Camera 1 Pedestrian Detection Mono Camera')
title('Camera 1 Pedestrian Detection Mono Camera with Semantic')
set(axes1,'FontSize', 40,'FontWeight','bold');

% Camera 2
FastPrecisionRaw2 = [0.0821147 0.114324 0.181034 0.276331 0.493976];
FastRecallRaw2 = [0.574238 0.565776 0.547045 0.469617 0.365588];

FastPrecisionSemantic2 = [0.578323 0.57955 0.613626 0.96997 0.967391];
FastRecallSemantic2 = [ 0.451123  0.451123 0.43428 0.213342 0.0293923];

figure1 = figure();
axes1 = axes('Parent',figure1);
plot(FastPrecisionRaw2, FastRecallRaw2, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(FastPrecisionSemantic2, FastRecallSemantic2, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('RAW', 'Semantic')
title('Camera 1 Pedestrian Detection Mono Camera')
title('Camera 1 Pedestrian Detection Mono Camera with Semantic')
set(axes1,'FontSize', 40,'FontWeight','bold');

% Camera 3
FastPrecisionRaw3 = [0.0840162 0.129406 0.273514 0.416787 0.576907];
FastRecallRaw3 = [0.759174 0.721339 0.662895 0.576 0.446631];

FastPrecisionSemantic3 = [0.686792 0.686792 0.717671 0.849528 0.708171];
FastRecallSemantic3 = [0.607071 0.607071 0.596064 0.419947 0.0607071];

figure1 = figure();
axes1 = axes('Parent',figure1);
plot(FastPrecisionRaw3, FastRecallRaw3, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(FastPrecisionSemantic3, FastRecallSemantic3, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('RAW', 'Semantic')
title('Camera 1 Pedestrian Detection Mono Camera')
title('Camera 1 Pedestrian Detection Mono Camera with Semantic')
set(axes1,'FontSize', 40,'FontWeight','bold');


% F-Scores
FScoreFastRaw1 = max((2 .* FastPrecisionRaw1 .* FastRecallRaw1)./(FastPrecisionRaw1 + FastRecallRaw1))
FScoreFastSemantic1 = max((2 .* FastPrecisionSemantic1 .* FastRecallSemantic1)./(FastPrecisionSemantic1 + FastRecallSemantic1))

FScoreFastRaw2 = max((2 .* FastPrecisionRaw2 .* FastRecallRaw2)./(FastPrecisionRaw2 + FastRecallRaw2))
FScoreFastSemantic2 = max((2 .* FastPrecisionSemantic2 .* FastRecallSemantic2)./(FastPrecisionSemantic2 + FastRecallSemantic2))

FScoreFastRaw3 = max((2 .* FastPrecisionRaw3 .* FastRecallRaw3)./(FastPrecisionRaw3 + FastRecallRaw3))
FScoreFastSemantic3 = max((2 .* FastPrecisionSemantic3 .* FastRecallSemantic3)./(FastPrecisionSemantic3 + FastRecallSemantic3))