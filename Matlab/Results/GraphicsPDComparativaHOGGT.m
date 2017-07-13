clc;
clear all;
close all;


%% Pedestrian Detection Mono Camera Semantic Constrains

% Camera 1
HOGPrecisionRaw1 = [0.578996 0.711222 0.884319 0.957447 1];
HOGRecallRaw1 = [0.218533 0.172983 0.0598053 0.00782337 0.000173853];

HOGPrecisionSemantic1 = [0.506849 0.630153 0.850394 0.944444 1];
HOGRecallSemantic1 = [0.115786 0.0930111 0.0375522 0.00591099 0.000173853];

figure1 = figure();
axes1 = axes('Parent',figure1);
plot(HOGPrecisionRaw1, HOGRecallRaw1, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(HOGPrecisionSemantic1, HOGRecallSemantic1, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('RAW', 'Semantic')
title('Camera 1 Pedestrian Detection Mono Camera')
title('Camera 1 Pedestrian Detection Mono Camera with Semantic')
set(axes1,'FontSize', 40,'FontWeight','bold');

% Camera 2
HOGPrecisionRaw2 = [0.169066 0.300659 0.400651 0.459016 0.75];
HOGRecallRaw2 = [0.182299 0.120542 0.0406209 0.00924703 0.000990753];

HOGPrecisionSemantic2 = [0.232962 0.413349 0.682081 0.756757 1];
HOGRecallSemantic2 = [0.172721 0.116579 0.0389696 0.00924703 0.000990753  ];

figure1 = figure();
axes1 = axes('Parent',figure1);
plot(HOGPrecisionRaw2, HOGRecallRaw2, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(HOGPrecisionSemantic2, HOGRecallSemantic2, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('RAW', 'Semantic')
title('Camera 1 Pedestrian Detection Mono Camera')
title('Camera 1 Pedestrian Detection Mono Camera with Semantic')
set(axes1,'FontSize', 40,'FontWeight','bold');

% Camera 3
HOGPrecisionRaw3 = [0.278007 0.413529 0.646532 0.734375 0.92];
HOGRecallRaw3 = [ 0.247333 0.197799 0.0963976 0.0313542 0.00767178 ];

HOGPrecisionSemantic3 = [0.305263 0.438356 0.683206 0.833333 1];
HOGRecallSemantic3 = [0.183728 0.138759 0.0597065 0.0183456 0.00433622];

figure1 = figure();
axes1 = axes('Parent',figure1);
plot(HOGPrecisionRaw3, HOGRecallRaw3, '-rx', 'MarkerSize', 10, 'LineWidth', 4)
hold on;
plot(HOGPrecisionSemantic3, HOGRecallSemantic3, '-gx', 'MarkerSize', 10, 'LineWidth', 4)
xlim([0 1]);ylim([0 1]);
xlabel('Precision','FontWeight','bold');
ylabel('Recall','FontWeight','bold');
legend('RAW', 'Semantic')
title('Camera 1 Pedestrian Detection Mono Camera')
title('Camera 1 Pedestrian Detection Mono Camera with Semantic')
set(axes1,'FontSize', 40,'FontWeight','bold');


% F-Scores
FScoreHOGRaw1 = max((2 .* HOGPrecisionRaw1 .* HOGRecallRaw1)./(HOGPrecisionRaw1 + HOGRecallRaw1))
FScoreHOGSemantic1 = max((2 .* HOGPrecisionSemantic1 .* HOGRecallSemantic1)./(HOGPrecisionSemantic1 + HOGRecallSemantic1))

FScoreHOGRaw2 = max((2 .* HOGPrecisionRaw2 .* HOGRecallRaw2)./(HOGPrecisionRaw2 + HOGRecallRaw2))
FScoreHOGSemantic2 = max((2 .* HOGPrecisionSemantic2 .* HOGRecallSemantic2)./(HOGPrecisionSemantic2 + HOGRecallSemantic2))

FScoreHOGRaw3 = max((2 .* HOGPrecisionRaw3 .* HOGRecallRaw3)./(HOGPrecisionRaw3 + HOGRecallRaw3))
FScoreHOGSemantic3 = max((2 .* HOGPrecisionSemantic3 .* HOGRecallSemantic3)./(HOGPrecisionSemantic3 + HOGRecallSemantic3))

