% Evaluacion GT vs detecciones propias
clear all;
close all;
clc;

video_dir = '/Users/alex/IPCV-MasterThesis/Matlab/Evaluation/Video';
PD_algorithm = {'FastRCNN'};
video_names = {'Camera1', 'Camera2', 'Camera3'};

for k = 1 : size(PD_algorithm,2)
    
    % Threshold vector
    thresholds = linspace(0, 1, 30);
    
    % General matrices
    Precision = zeros(size(thresholds,2), size(video_names,2));
    Recall    = zeros(size(thresholds,2), size(video_names,2));
    F1Score   = zeros(size(thresholds,2), size(video_names,2));
    
    fp = zeros(size(thresholds,2), size(video_names,2));
    tp = zeros(size(thresholds,2), size(video_names,2));
    fn = zeros(size(thresholds,2), size(video_names,2));
    
    for i = 1 : size(video_names,2)
        
        processing_mask.x = 0;
        processing_mask.y = 0;
        processing_mask.w = 640;
        processing_mask.h = 480;
        
        % Files paths
        filename_gt = sprintf('%s/Pedestrian Detections/Camera%dGT.txt', video_dir, i);
        filename_1 = sprintf('%s/Pedestrian Detections/BoundingBoxes %s %d.idl', video_dir, PD_algorithm{k}, i);
        
        for j = 1 : size(thresholds,2)
            if j == 1
                % Get GT and PD blobs
                [GTBlobs] = ReadGTBlobs(filename_gt, video_names{i}, processing_mask);
                [Blobs_total_people, Scores_total_people] = ReadPDBlobs(filename_1, video_names{i}, processing_mask);
                
                % Score histogram representation
                figure(k + 20)
                subplot(3, 1, i)
                histogram(Scores_total_people,30)
                xlim([0 1]);
                title([PD_algorithm{k} ' Score histogram ' video_names{i}])
            end
            
            % Score thresholding
            PDBlobs = Blobs_Threshold(Blobs_total_people, thresholds(j));
            
            num_frame_gt = size(GTBlobs,2);
            num_frame_PD = size(PDBlobs,2);
            num = [num_frame_gt num_frame_PD];
            num_frames = min(num);
            
            % Extract measures
            [Blobs, Precision(j,i), Recall(j,i), F1Score(j,i), labels, scores, tp(j,i), fp(j,i), tn, fn(j,i), num_blobs_gt_out(i)] = PeopleDetectionEval(GTBlobs, PDBlobs, num_frames);
            display(['Algoritmo: ' PD_algorithm{k} ' - ' video_names{i} ' - Threshold: ' num2str(j) '/' num2str(size(thresholds,2))]);
        end
        
        % Vectors for the actual curve
        PrecisionGraph = [0 Precision(:,i)'];
        RecallGraph = [Recall(1,i) Recall(:,i)'];
        FScore = 2 * (PrecisionGraph .* RecallGraph) ./ (PrecisionGraph + RecallGraph);
        
        aux = 0;
        ejey = RecallGraph;
        ejex = 1 - PrecisionGraph;
        ejex = fliplr(ejex);
        ejey = fliplr(ejey);
        
        for n = 1 : size((ejex), 2) - 1
            aux = aux + ((ejey(n) + ejey(n+1))) / 2 * (ejex(n + 1) - ejex(n));
        end
        AUC(i) = aux;
        
        % Representation of curves
        figure(k)
        plot(RecallGraph, PrecisionGraph)
        hold on
        axis([0 1 0 1])
        xlabel('Recall')
        ylabel('Precision')
        title([PD_algorithm{k} ' Precision/Recall'])
        legend('Camera 1', 'Camera 2', 'Camera 3')
    end
    disp(['Camera 1 AUC: ' num2str(AUC(1))]);
    disp(['Camera 2 AUC: ' num2str(AUC(2))]);
    disp(['Camera 3 AUC: ' num2str(AUC(3))]);
    
    % Saving
    % save(sprintf('%s_%.2f_ams2_gtvoc_nms2.mat',model_name_out{k},threshold), 'AUC_dtdp', 'AUC_dtdp_mean_1', 'F1Score_dtdp', 'Precision_dtdp', 'Recall_dtdp', 'fp_dtdp', 'fscore_dtdt', 'precision_dtdp', 'recall_dtdp', 'threshols', 'threshols', 'tp_dtdp','fn_dtdp');
end
