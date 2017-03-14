function [ ProposalMGCFixed ] = ProposalExtractor( Image, Method )

% OBJECT PROPOSAL EXTRACTION
% MGC
[proposals, ~, ~] = im2mcg(Image, Method);
% [y1, x1, y2, x2]
ProposalMGC = proposals.bboxes;

% Change the order of the coordinates to be
% [x1, y1, x2, y2]
ProposalMGCFixed(:,1) = ProposalMGC(:,2);
ProposalMGCFixed(:,2) = ProposalMGC(:,1);
ProposalMGCFixed(:,3) = ProposalMGC(:,4);
ProposalMGCFixed(:,4) = ProposalMGC(:,3);

end

