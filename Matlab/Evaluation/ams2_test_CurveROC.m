% Evaluaci?n GT vs detecciones propias

clear all;
close all;
clc;

% video_dir = '/Users/AlejandroMiguelez/Desktop/Experimento Nuevo/Finales';
video_dir = '/Users/alex/IPCV-MasterThesis/Matlab/Evaluation/Video';

video_names = {''};
% video_cams = {'cam0'};
video_cams = {'cam1'};
threshold=-5;

model_name_people = {'inriaperson_final'};
model_name_out = {'inriaperson_final_EN'};

for k = 1:size(model_name_people,2)

% pd_sp = load(sprintf(['configurations_final_-4_dkl_32bits.mat']));

threshols = [-2.5:0.6:5];

Precision_dtdp = zeros(size(threshols,2), size(video_names,2));
Recall_dtdp = zeros(size(threshols,2), size(video_names,2));
F1Score_dtdp = zeros(size(video_names,2));
fp_dtdp = zeros(size(threshols,2), size(video_names,2));
tp_dtdp = zeros(size(threshols,2), size(video_names,2));


for i = 1:size(video_names,2)
    
%   processing_mask.x = 0;
% 	processing_mask.y = 0;
% 	processing_mask.w = 768;
% 	processing_mask.h = 576;
    processing_mask.x = 0;
	processing_mask.y = 0;
	processing_mask.w = 480;
	processing_mask.h = 270;
    
    filename_gt = sprintf('%s/cam0/cam0_gt.txt', video_dir);
% 	filename_dtdp_1 = sprintf('%s/cam0/cam0_dtdp_parts_original_inria_thr-2.50.idl', video_dir);
    filename_dtdp_1 = sprintf('%s/cam0/cam1_dtdp_parts_cam0_hcorr_-2.50.idl', video_dir);
                                       
    for j=1:size(threshols,2)
        if j==1
            [Blobs_gt] = ReadGTBlobs(filename_gt, video_names{i}, processing_mask);

            [Blobs_dtdp_total_people, ~] = ReadDTDPMBlobs3(filename_dtdp_1, video_cams{j}, processing_mask);
            
%             for ii=1:length(Blobs_gt)
%                 Blobs_dtdp_total_people2{ii}=Blobs_dtdp_total_people{ii};
%             end
            
%             Blobs_dtdp_total_people=Blobs_dtdp_total_people2;
%             for ii=1:length(Blobs_dtdp_total_people)
%                 for jj=1:length(Blobs_dtdp_total_people{ii})
%                     Blobs_dtdp_total_people{ii}(jj).score=(pd_sp.PD_original,Blobs_dtdp_total_people{ii}(jj).score);
%                 end
%             end
        end
        
        Blobs_dtdp = Blobs_Threshold(Blobs_dtdp_total_people, threshols(j));
        
        num_frame_gt = size(Blobs_gt,2);
        num_frame_dtdp = size(Blobs_dtdp,2);       
        num = [num_frame_gt num_frame_dtdp];
        num_frames = min(num);
                 
        [Blobs,Precision_dtdp(j,i),Recall_dtdp(j,i),F1Score_dtdp(j,i),labels,scores,tp_dtdp(j,i),fp_dtdp(j,i),tn,fn_dtdp(j,i),num_blobs_gt_out(i)]=PeopleDetectionEval(Blobs_gt,Blobs_dtdp,num_frames);  
        display(['Secuencia: ' video_names{i} ' - Frame(gtvsc): ' num2str(j)]);
    end
    
    precision_dtdp = [0 Precision_dtdp(:,i)'];
    recall_dtdp = [Recall_dtdp(1,i) Recall_dtdp(:,i)'];
    fscore_dtdt = 2*(precision_dtdp.*recall_dtdp)./(precision_dtdp+recall_dtdp);
    
    aux = 0;
    ejey = recall_dtdp;
    ejex = 1-precision_dtdp;
    ejex = fliplr(ejex);
    ejey = fliplr(ejey);
    
    for n = 1:size((ejex),2)-1
        aux = aux+((ejey(n)+ejey(n+1)))/2*(ejex(n+1)-ejex(n));
    end
    AUC_dtdp(i)=aux;
    plot(recall_dtdp,precision_dtdp)
    axis([0 1 0 1])
    xlabel('Recall')
    ylabel('Precision')
end

AUC_dtdp_mean_1 = mean(AUC_dtdp)
% save(sprintf('%s_%.2f_ams2_gtvsc_nms2.mat',model_name_out{k},threshold), 'AUC_dtdp', 'AUC_dtdp_mean_1', 'F1Score_dtdp', 'Precision_dtdp', 'Recall_dtdp', 'fp_dtdp', 'fscore_dtdt', 'precision_dtdp', 'recall_dtdp', 'threshols', 'threshols', 'tp_dtdp','fn_dtdp');
save(sprintf('%s_%.2f_ams2_gtvoc_nms2.mat',model_name_out{k},threshold), 'AUC_dtdp', 'AUC_dtdp_mean_1', 'F1Score_dtdp', 'Precision_dtdp', 'Recall_dtdp', 'fp_dtdp', 'fscore_dtdt', 'precision_dtdp', 'recall_dtdp', 'threshols', 'threshols', 'tp_dtdp','fn_dtdp');
end
