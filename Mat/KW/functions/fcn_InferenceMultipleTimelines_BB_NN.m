function [node] = fcn_InferenceMultipleTimelines_BB_NN(s_task,deadline_task,length_task,drop_task,w_task,N,net)




% Policy Neural Net Implementation
% tic
PF(1,:) = s_task;
PF(2,:) = deadline_task;
PF(3,:) = length_task; % Also called d_task elsewhere
PF(4,:) = drop_task;
PF(5,:) = w_task;

PFtree = zeros(N,N);
PfStatus = zeros(3,N);
PfStatus(1,:) = 1;
%         for nn = 1:length(node)
%             PfStatus(: , node(nn) ) = [0; 0; 1]; % Infeasible Already Assigned
%         end

Xin = [PF; PFtree; PfStatus];
node = zeros(N,1);
for kk = 1:N
    [YPred,scores] = classify(net,Xin);
    scores(node(node ~= 0)) = 0;
    scores = scores/(sum(scores));
    [~,YPred] = max(scores);
    node(kk) = double(YPred);
    
    PFtree = zeros(N,N);
    IND = sub2ind([N N],[1:kk]',node(1:kk));
    PFtree(IND) = 1;
    
    PfStatus = zeros(3,N);
    PfStatus(1,:) = 1;
    for nn = 1:kk
        PfStatus(: , node(nn) ) = [0; 0; 1]; % Infeasible Already Assigned
    end
    Xin = [PF; PFtree; PfStatus];
    
end


% 
% [Cost.NN(monte,cnt.N),t_ex,NumDropTask] = MultiChannelSequenceScheduler(node,N,K,s_task,w_task,deadline_task,length_task,drop_task);
% DropPercent.NN(monte,cnt.N) = NumDropTask/N;
% RunTime.NN(monte,cnt.N) = toc;