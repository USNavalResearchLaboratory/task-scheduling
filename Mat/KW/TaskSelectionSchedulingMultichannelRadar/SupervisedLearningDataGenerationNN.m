function [X,Y] = SupervisedLearningDataGenerationNN(NodeStats,NodeParams,K,CutOff)

% Generate Data X and labels Y for training of Cognitive Resource Manager

% Inputs
% NodeStats: Contains B&B sequences, associated optimal sequence from root
% node and associated costs
%
% NodeParams: Inputs to B&B (Used as features for Supervised Learner)

% Outputs
% X - feature data
% Y - labels (optimal actions)


% Feature Data
s_task = NodeParams.s_task;
deadline_task = NodeParams.deadline_task;
length_task = NodeParams.length_task;
drop_task = NodeParams.drop_task;
w_task = NodeParams.w_task;
refTime = min(s_task);

% Generate Policy Features
N = length(s_task);
PF(1,:) = s_task - refTime; % Make first feature relative to minimum start time KW - 09APR2020
PF(2,:) = deadline_task - refTime; % Make 2nd feature relative to minimum start time KW - 09APR2020
% PF(1,:) = s_task;
% PF(2,:) = deadline_task;
PF(3,:) = length_task;
PF(4,:) = drop_task;
PF(5,:) = w_task;


[Cost,sortIdx] = sort([NodeStats.BestCost]); % Organize Costs in Ascending order
OptimalSeq = NodeStats(sortIdx(1)).BestSeq;

sortIdx(CutOff+1:end) = [];


% Expand Children of All NodeStats
cnt = 1;
RecordSeq = zeros(N,1);
CostFinal = 0;
zeroVec = zeros(N,1);

for jj = sortIdx
    
    curNode = NodeStats(jj);
    BestSeq = curNode.BestSeq;
    for kk = 1:N
        
        node = BestSeq(1:kk-1);
        optimal_action = BestSeq(kk);
        T = [node];
        new_seq = [T; zeros(N-length(T),1)];
        
        if cnt == 1
            VisitIndicator = 0;
        else
            SeqDiff = ( RecordSeq - new_seq );
%             SeqDiff = bsxfun(@plus, RecordSeq, -new_seq );
%             VisitIndicator = sum( sum( SeqDiff == zeroVec , 1) == N );
            VisitIndicator = sum(~any(SeqDiff));            
%             if VisitIndicator ~= VisitIndicator2
%                 keyboard
%             end            
        end
        
        if VisitIndicator == 0            
            RecordSeq( :  , cnt) = new_seq;
            CostFinal(cnt) = curNode.BestCost;            
            Y(cnt,1) = optimal_action;
            
            PFtree = zeros(N,N);
            IND = sub2ind([N N],[1:length(node)]',node);
            PFtree(IND) = 1;            
            
            PfStatus = zeros(3,N);
            PfStatus(1,:) = 1;
            for nn = 1:length(node)
                PfStatus(: , node(nn) ) = [0; 0; 1]; % Infeasible Already Assigned 
            end

            X(:,:,cnt) = [PF; PfStatus; PFtree; ]; % Timeline ignored for now
            
            % Note: Could find if node is dominated by calculating cost at
            % the partial sequence and comparing across other partial
            % nodes. Not doing at the moment.
            
            cnt = cnt + 1;
        end        
    end
end


%% Old Method which doesn't do comparison across different tree branches. Suboptimal and may have some errors in the code. Results in confusing labels.
% For example suppose [1 2 3] is best --> the code below may generate
% mutliple root node actions [0 0 0] optimal action is 1 overall but if
% you're in another branch it initials the root nodes optimal action as 2
% or 3 depending on that branch. Results in garbage input to NN.
% Hopefully this makes sense to us in the future KW 1/24/20.



% 
% % Y = categorical;
% RecordSeq = zeros(N,1);
% cnt = 1;
% for jj = sortIdx
%     
%     curNode = NodeStats(jj);
%     BestSeq = curNode.BestSeq;
%     
%     
%     InputTimeLine = zeros(K,1);
%     for kk = 1:N
%         
%         node = BestSeq(1:kk-1);
%         optimal_action = BestSeq(kk);
%         T = [node; optimal_action];        
%         new_seq = [T; zeros(N-length(T),1)];
%         
%         ChannelAvailableTime = zeros(K,1); % Note BBMultiChannelSequenceScheduler assigns entire sequence therefore need available time to be all zeros initially
%         [t_ex,x,OutputTimeLine,TaskChannel] = BBMultiChannelSequenceScheduler(T,s_task,deadline_task,length_task,ChannelAvailableTime);
%         
%         
%         SeqDiff = ( RecordSeq - new_seq );
%         
%         VisitIndicator = sum( sum( SeqDiff == [0; 0; 0], 1) == N );  % If VisitIndicator is 0, then you haven't visited it yet  
% %         VisitIndicator = sum( sum( RecordSeq - new_seq ) == 0 );        
%         if VisitIndicator == 0 
%             
%             RecordSeq( :  , cnt) = new_seq;
%                                    
%             PFtree = zeros(N,N);
%             IND = sub2ind([N N],[1:length(node)]',node);
%             PFtree(IND) = 1;            
%             
%             PfTimeLine = repmat(InputTimeLine,1,N);
%             PfStatus = zeros(3,N);
%             PfStatus(1,:) = 1;
%             for nn = 1:length(node)
%                 PfStatus(: , node(nn) ) = [0; 0; 1]; % Infeasible Already Assigned 
%             end
% 
%             X(:,:,cnt) = [PF; PfStatus; PFtree; ]; % Timeline ignored for now
% %             X(:,:,cnt) = [PF; PfTimeLine; PFtree; ]; % Put tree at bottom to allow conv. kernel to see both PF and timeline easier
%             Y(cnt,1) = optimal_action; % In this version only assign task. Channel assignment implemented by separate function
%                     
%             % If you want to jointly assign task and channel 
% %             if TaskChannel(optimal_action) == 0 % Drop the task
% %                 % Could potentially just assign to timeline 1 and handle
% %                 % this case in logic later
% %                 Y(cnt,1) = N*K + 1; % Additional Class to indicate dropping (i.e. tasks is not assigned to a timeline)
% %             else            
% %                 Y(cnt,1) = optimal_action + (TaskChannel(optimal_action) - 1)*N; % Additional Classes to capture assignment to K timelines K*N possible actions
% %             end
% %             if Y(cnt,1) <= 0
% %                 keyboard
% %             end
%             cnt = cnt + 1;          
%         end
%         InputTimeLine = OutputTimeLine;        
%         
%     end        
% end





