function [Cost,t_ex,NumDropTask,T,ChannelAvailableTime] = MctsNeuralNetPureSchedulerAlgorithm(data,net,MONTE)

% Earliest Deadline Algorithm
% Takes tasks in data and assigns them to timeline using the Earliest deadline.
% Input: data summarized below
% net - neural net used for transition probabilities
% MONTE - number of roll-out per node

% Output:
% Cost - cost of assigning tasks to timeline using this algorithm
% t_ex - time tasks are executed
% T - sequence of tasks T(1) is first task assigned, T(N) last task
% NumDropTask - number of tasks that get dropped

N = data.N; % Number of tasks
K = data.K; % Number of timelines
s_task = data.s_task; % Start times (release-times) of tasks
w_task = data.w_task; % Weights of tasks. Bigger --> higher priority
deadline_task = data.deadline_task; % When task will be dropped
length_task = data.length_task; % How long tasks takes to complete
drop_task = data.drop_task; % Penalty for dropping task
RP = data.RP;
ChannelAvailableTime = data.ChannelAvailableTime;

%% MCTS
N = length(s_task);
% Initialize Stack

UB = inf; % Upper Bound
S = struct('T',[],'ND',(1:N)','D',[],'CompleteSolutionFlag',0,'BestCost',Inf,'BestSeq',[]);
%     't_ex',zeros(N,1),'ChannelAvailableTime',zeros(K,1),'ChannelAssignment',zeros(N,1),'x',zeros(N,1),...
%     'CompleteSolutionFlag',0,'BestCost',Inf,'BestSeq',[]);
% Snew = S;
% S.T = [];   % Task sequence
% S.PF = (1:N)'; % Possible First Task
% S.NS = [];      % Not scheduled task
% S.DR = [];      % Dropped task
% S.ND = [];      % Not dominated Set
%
% % NodeStats contains relevant node statistics resulting from B&B search
% NodeStats = struct('T',[],'PF',(1:N)','NS',[],'DR',[],'ND',[], ...
%     't_ex',zeros(N,1),'ChannelAvailableTime',zeros(K,1),'ChannelAssignment',zeros(N,1),'x',zeros(N,1),...
%     'CompleteSolutionFlag',0,'BestCost',Inf,'BestSeq',[]);
% % NodeStats = struct('T',[],'Tterminal',[],'NS',[],'ND',[],'CompleteSolutionFlag',0,'Cost',Inf);
NodeCnt = 1; % Counter of number recorded nodes


% Intialize Channel Availability Tracker
% ChannelAvailableTimeInput = zeros(K,1);

% new_node = 1;
while ~isempty(S.ND) % Proceed if base-node ND set is not empty
    
    for monte = 1:MONTE
        Sprime = S;
        Tprime = Sprime.T;
        NDprime = Sprime.ND;
        Dprime = Sprime.D;
        Np = net.Layers(1).InputSize(2); % Size of Neural Net input
        
        while ~isempty(NDprime)
            
            % Expand
            if length(NDprime) >= Np % Take all tasks from ND set
                [~,task_index] = sort(s_task(NDprime));
                task_index = NDprime(task_index(1:Np));      % START HERE 
            else
                task_index = NDprime;
                task_index = [Tprime; NDprime ];
            end            
            %         InputIndex = [Tprime NDprime(1:
            
            % Encode Features
            PF(1,:) = [s_task(task_index)  ; zeros(Np-length(task_index),1)] ;
            PF(2,:) = [deadline_task(task_index); zeros(Np-length(task_index),1)] ;
            PF(3,:) = [length_task(task_index); zeros(Np-length(task_index),1)];
            PF(4,:) = [drop_task(task_index); zeros(Np-length(task_index),1)];
            PF(5,:) = [w_task(task_index); zeros(Np-length(task_index),1)];
            
            % Encode Position in Tree
            PFtree = zeros(Np,Np);
            TreeIndex = [];
            for jj = 1:length(Tprime)
                if ~isempty( find(task_index == Tprime(jj))  )
                    TreeIndex(jj,1) = find(task_index == Tprime(jj));
                end
            end
            if ~isempty(TreeIndex)
                IND = sub2ind([Np Np],[1:length(TreeIndex)]',TreeIndex);            
                PFtree(IND) = 1;
            end
            
            % Encode whether node was assigned/dominated/available
            PfStatus = zeros(3,Np);
            PfStatus(1,:) = 1;                    
            for nn = 1:length(TreeIndex)
                PfStatus(: , TreeIndex(nn) ) = [0; 0; 1]; % Infeasible Already Assigned
            end
            
            
            Xin = [PF; PFtree; PfStatus];
            node = zeros(Np,1);
            node(1:length(Tprime)) = Tprime;
            for kk = (length(Tprime)+1):Np         
                
                % Generate aprior Probabilities
                [YPred,scores] = classify(net,Xin);
                scores(TreeIndex(TreeIndex ~= 0)) = 0; % Remove Nodes from Scoring that have been assigned
                if ~isempty(Tprime)
                    for jj = 1:length(Tprime)
                        scores( task_index == Tprime(jj) ) = 0; % Remove nodes when number of inputs < NN capacity
                    end
                end
                scores = scores/(sum(scores));
                apriori = scores;
                
                % Selection
                RN = rand;
                TempIndex = find( RN > cumsum(apriori),1,'last');
                if isempty(TempIndex)
                    SelectionIndex = task_index(1);
                else
                    try 
                    SelectionIndex = task_index(TempIndex(end) + 1);
                    catch
                        keyboard
                    end
                end
                
                Tprime = [Tprime; SelectionIndex];
                NDprime(NDprime == SelectionIndex) = [];
                
                % Generate Image for next iteraion
                node(kk) = SelectionIndex;
                for jj = 1:length(Tprime)
                    if ~isempty( find(task_index == Tprime(jj))  )
                        TreeIndex(jj,1) = find(task_index == Tprime(jj));
                    end
                end
                if ~isempty(TreeIndex)
                    IND = sub2ind([Np Np],[1:length(TreeIndex)]',TreeIndex);
                    PFtree(IND) = 1;
                end
                
                % Encode whether node was assigned/dominated/available
                PfStatus = zeros(3,Np);
                PfStatus(1,:) = 1;
                for nn = 1:length(TreeIndex)
                    PfStatus(: , TreeIndex(nn) ) = [0; 0; 1]; % Infeasible Already Assigned
                end
                
                
%                 PFtree = zeros(Np,Np);
%                 IND = sub2ind([Np Np],[1:kk]',node(1:kk));
%                 PFtree(IND) = 1;
%                 
%                 PfStatus = zeros(3,Np);
%                 PfStatus(1,:) = 1;
%                 for nn = 1:kk
%                     PfStatus(: , node(nn) ) = [0; 0; 1]; % Infeasible Already Assigned
%                 end
                Xin = [PF; PFtree; PfStatus];
                               
                
            end
            
        end
        
        [Cprime(monte)] = MultiChannelSequenceScheduler(Tprime,N,K,s_task,w_task,deadline_task,length_task,drop_task);
        SeqMonte(:,monte) = Tprime;
        
%         node = zeros(N,1);
%         for kk = 1:N
%             [YPred,scores] = classify(net,Xin);
%             scores(node(node ~= 0)) = 0;
%             scores = scores/(sum(scores));
%             [~,YPred] = max(scores);
%             node(kk) = double(YPred);
%             
%             PFtree = zeros(N,N);
%             IND = sub2ind([N N],[1:kk]',node(1:kk));
%             PFtree(IND) = 1;
%             
%             PfStatus = zeros(3,N);
%             PfStatus(1,:) = 1;
%             for nn = 1:kk
%                 PfStatus(: , node(nn) ) = [0; 0; 1]; % Infeasible Already Assigned
%             end
%             Xin = [PF; PFtree; PfStatus];
%         end
        
                %         [~,task_index] = min(s_task(NDprime));
                
    end
    
    % Make a Decision
    [Cbest,minIndex ] = min(Cprime);
    Seq = SeqMonte(:,minIndex);    
    if Cbest < UB
        UB = Cbest;
    end
    
    %  Create new base-node
    S.CompleteSolutionFlag = 1;
    S.BestCost = Cbest;
    S.BestSeq = Seq;
    Depth = length(S.T) + 1;
    S.T = Seq(1:Depth);
    S.ND = setdiff(S.ND,S.T);
    
    
    
    
    
end

%% Evaluate the Cost and pass out the sequence
T = S.T;

if ~strcmpi(data.scheduler,'flexdar')    
    [Cost,t_ex,NumDropTask] = FunctionMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task,ChannelAvailableTime);
else
    [Cost,t_ex,ChannelAvailableTime,NumDropTask] = FlexDARMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task,ChannelAvailableTime,RP);
end




%% Old Stuff
%
%
%     % Pull off data from end of stack
%     %     UB = S(end).UB;
%     PF = S(end).PF;
%     T = S(end).T;
%     NS = S(end).NS;
%     DR = S(end).DR;
%     ND = S(end).ND;
%     TimeExecutionInput = S(end).t_ex;
%     ChannelAvailableTimeInput = S(end).ChannelAvailableTime;
%     ChannelAssignmentInput = S(end).ChannelAssignment;
%     ScheduledIndicatorInput = S(end).x; % x indicates whether channel has been scheduled (1) or not scheduled (0)
%
%     %     [t_ex,x,ChannelAvailableTimeProposed,TaskChannel] = BBMultiChannelSequenceScheduler(T,s_task,deadline_task,length_task,ChannelAvailableTimeInput);
%     %     Cprime = sum( w_task(T).*(t_ex(T) - s_task(T)) )  + sum( drop_task(DR) ) ;
%     %     Cupper = sum( w_task(T).*(t_ex(T) - s_task(T)) )  + sum( drop_task(PF) ) ;
%
%
%
%     %     if Cprime < UB
%
%
%     if ~isempty(PF)
%
%         [~,task_index] = min(s_task(PF));
%
%         if all( s_task(task_index) == s_task) % All of the start times are the same. Sort by weights
%             [~,task_index] = max(w_task(PF));
%         end
%
%
%
%         %             if new_node
%         %                 new_node = 0;
%         %             else
%         %                 task_index = randi(length(PF),1); % Randomly choose a task to schedule
%         %             end
%         task = PF(task_index);
%         PF(task_index) = [];
%         Tprime = [T; task];
%         PFprime  = [PF; NS];
%         NSprime = [];
%         DRprime = DR;
%         NDprime = [];
%         NS = [NS; task];
%
%         % Update S(end)
%         S(end).PF = PF;
%         S(end).NS = NS;
%
%         % Update Schedule Parameters After adding latest task
%         %             x_input = (t_ex < deadline_task);
%         [t_ex,x,ChannelAvailableTimeProposed,TaskChannel] = BBUpdateSequenceParameters(Tprime,s_task,deadline_task,length_task,ChannelAvailableTimeInput,TimeExecutionInput,ChannelAssignmentInput,ScheduledIndicatorInput);
%
%         %             [t_ex,x,ChannelAvailableTimeProposed,TaskChannel] = BBMultiChannelSequenceScheduler(Tprime,s_task,deadline_task,length_task,ChannelAvailableTimeInput);
%
%
%         %                 [~,t_ex,~] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
%         ExecutionTimeOffsetFlag = diff([0; t_ex(Tprime)]) >= 0 ;
%         if all(ExecutionTimeOffsetFlag) % Start Times domainance rule
%
%             % Move any task whose deadline has passed on all
%             % timelines from PFprime to DRprime
%             deadline_indicator = zeros(N,K);
%             for kk = 1:K
%                 deadline_indicator(:,kk) =   deadline_task <  ChannelAvailableTimeProposed(kk);
%             end
%             pass_index = find( sum(deadline_indicator(PFprime,:),2) == K );
%             if ~isempty(pass_index) % Move tasks from PFprime to DRprime
%                 DRprime = [DRprime; PFprime(pass_index)];
%                 PFprime(pass_index) = [];
%                 %                     keyboard
%             end
%             % Compute Partial Schedule Cost
%             Cprime = sum( w_task(Tprime).*(t_ex(Tprime) - s_task(Tprime)) )  + sum( drop_task(DRprime) ) ;
%
%             % Check if schedule is active (PFprime checked to see
%             % if any can task be scheduled right before last task
%             % in Tprime on same timeline without imposing a delay
%             active_flag = 1;
%             PartialSchedule = Tprime(1:end-1);
%             ChannelSelected = TaskChannel(Tprime(end));
%             %                 [t_partial,x_partial,ChannelAvailableTimePartial,PartialChannelAssignment] = BBMultiChannelSequenceScheduler(PartialSchedule,s_task,deadline_task,length_task,zeros(K,1));
%             %                 [t_partial,x_partial,ChannelAvailableTimePartial,PartialChannelAssignment] = BBActiveSequenceScheduler(PartialSchedule,s_task,deadline_task,length_task,zeros(K,1),ChannelSelected);
%
%
%
%             t_partial = TimeExecutionInput;
%             ChannelAvailableTimePartial = ChannelAvailableTimeInput;
%             PartialChannelAssignment = ChannelAssignmentInput;
%             x_partial = ScheduledIndicatorInput;
%
%             if sum(ChannelAvailableTimeInput - ChannelAvailableTimePartial) ~= 0
%                 keyboard
%             end
%
%
%
%             for jj = 1:length(PFprime)
%                 curTask = PFprime(jj);
%                 Tactive = [PartialSchedule; curTask; Tprime(end)];
%                 %                     ChannelSelected = TaskChannel(Tprime(end));
%
%                 [t_active] = BBPartialSequenceScheduler(t_partial,x_partial,Tactive,s_task,deadline_task,length_task,ChannelAvailableTimePartial,PartialChannelAssignment,ChannelSelected);
%
%
%                 %                     [t_active] = BBActiveSequenceScheduler(Tactive,s_task,deadline_task,length_task,ChannelAvailableTimeInput,ChannelSelected);
%
%
%                 if t_active(Tprime(end)) <= t_ex(Tprime(end))   %( s_task(curTask) + length_task(curTask) ) < t_ex(Tprime(end))
%                     active_flag = 0; %
%                     %                     keyboard
%                     break
%                 end
%             end
%
%
%
%
%             if active_flag
%
%                 if isempty(PFprime)
%                     S(end).CompleteSolutionFlag = 1;
%                     if Cprime < S(end).BestCost
%                         S(end).BestCost = Cprime;
%                         S(end).BestSeq = [Tprime; DRprime];
%                     end
%                 else
% %                     keyboard
%
%                     CurSeqBest = S(end).BestSeq;
%                     SeqBest = [Tprime; PFprime; DRprime];
%
%                     if isempty(CurSeqBest)
%                         CurSeqBest = zeros(N,1);
%                     end
%                     deltaSeq = SeqBest - CurSeqBest;
%
%                     if any(deltaSeq)
%
%                         if length(SeqBest) ~= N
%                             keyboard
%                         end
%                         [Cbest] = MultiChannelSequenceScheduler(SeqBest,N,K,s_task,w_task,deadline_task,length_task,drop_task);
%                         if Cbest < S(end).BestCost
%                             S(end).BestCost = Cbest;
%                             S(end).BestSeq = SeqBest;
%                         end
%                         % Assume All tasks of PFprime are dropped and update
%                         % terminal cost accordingly (need to do this because
%                         % lows-active flag is not being used
%                     else
%                         Cbest = S(end).BestCost;
%                     end
%
%                 end
%
%                 if Cprime < UB
%
%                     % Update ND set
%                     ND = ([ND; task]);
%                     %                     NodeStats(end).ND = ND; % Update node statistics
% %                     if length(ND) > 1
% %                         keyboard
% %                     end
%
%                     S(end).ND = ND;
%
%                     %                         LowsFlag = LOWS_active(K,Tprime,PFprime,DRprime,TaskChannel,s_task,w_task,deadline_task,drop_task,length_task,ChannelAvailableTimeInput);
%                     %                         if LowsFlag  % Not implemented correctly. Just
%                     %                         brute force w/o checking this step.
%
%                     % Push (Tprime,PFprime,NSprime,DRprime) onto
%                     % stack
%                     %                     Snew.UB = inf;
%                     Snew.T = Tprime;
%                     Snew.PF = PFprime;
%                     Snew.NS = NSprime;
%                     Snew.DR = DRprime;
%                     Snew.ND = NDprime;
%                     Snew.t_ex = t_ex;
%                     Snew.x = x;
%                     Snew.ChannelAvailableTime = ChannelAvailableTimeProposed;
%                     Snew.ChannelAssignment = TaskChannel;
%                     Snew.BestSeq = SeqBest;
%                     Snew.BestCost = Cbest;
%
%
%                     S = [S; Snew];
%                     %                             new_node = 1;
%                     %                         end
%                 end
%
%             end
%
%
%         end
%     else
%
%         %             keyboard
%
%
%
%
% %         [t_ex,x,ChannelAvailableTimeProposed,TaskChannel] = BBMultiChannelSequenceScheduler(T,s_task,deadline_task,length_task,zeros(K,1));
%
%         t_ex = TimeExecutionInput;
%
% %         if ~all(TimeExecutionInput == t_ex)
% %             keyboard
% %         end
%
%
%         C = sum( w_task(T).*(t_ex(T) - s_task(T)) )  + sum( drop_task(DR) ) ;
%         if isempty(NS) && C < UB
%             UB = C;
%             Tstar = T;
%             Tdr = DR;
%             Tfinal = [T; DR];
%         end
%         if S(end).CompleteSolutionFlag == 1
%             %             S(end).CompleteSolutionFlag = 1;
%             if C < S(end).BestCost && length(T) == N
%                 S(end).BestCost = C;
%                 S(end).BestSeq = T;
%                 keyboard
%             end
%             NodeStats(NodeCnt) = S(end);
%             NodeCnt = NodeCnt + 1;
%             %             keyboard
%         end
%         S(end) = [];
%
%
%     end
%
%     %     else
%     %         S(end) = [];
%     %     end
%
% end
%
%
% fields = {'t_ex','ChannelAvailableTime','ChannelAssignment','x'};
% NodeStats = rmfield(NodeStats,fields);




%%
% PF(1,:) = s_task;
% PF(2,:) = deadline_task;
% PF(3,:) = length_task;
% PF(4,:) = drop_task;
% PF(5,:) = w_task;
% 
% PFtree = zeros(N,N);
% PfStatus = zeros(3,N);
% PfStatus(1,:) = 1;
% %         for nn = 1:length(node)
% %             PfStatus(: , node(nn) ) = [0; 0; 1]; % Infeasible Already Assigned
% %         end
% 
% Xin = [PF; PFtree; PfStatus];
% node = zeros(N,1);
% for kk = 1:N
%     [YPred,scores] = classify(net,Xin);
%     scores(node(node ~= 0)) = 0;
%     scores = scores/(sum(scores));
%     [~,YPred] = max(scores);
%     node(kk) = double(YPred);
%     
%     PFtree = zeros(N,N);
%     IND = sub2ind([N N],[1:kk]',node(1:kk));
%     PFtree(IND) = 1;
%     
%     PfStatus = zeros(3,N);
%     PfStatus(1,:) = 1;
%     for nn = 1:kk
%         PfStatus(: , node(nn) ) = [0; 0; 1]; % Infeasible Already Assigned
%     end
%     Xin = [PF; PFtree; PfStatus];
% end
% 
% [Cost,t_ex,NumDropTask] = MultiChannelSequenceScheduler(node,N,K,s_task,w_task,deadline_task,length_task,drop_task);
% T = node;
