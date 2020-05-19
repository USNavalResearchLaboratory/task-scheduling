function [Tfinal,Tstar,Tdr,NodeStats] = BBschedulerWithStats(K,s_task,deadline_task,length_task,drop_task,w_task)


N = length(s_task);
% Initialize Stack

UB = inf; % Upper Bound
S = struct('T',[],'PF',(1:N)','NS',[],'DR',[],'ND',[], ...
    't_ex',zeros(N,1),'ChannelAvailableTime',zeros(K,1),'ChannelAssignment',zeros(N,1),'x',zeros(N,1),...
    'CompleteSolutionFlag',0,'BestCost',Inf,'BestSeq',[]);
Snew = S;

S.T = [];   % Task sequence
S.PF = (1:N)'; % Possible First Task
S.NS = [];      % Not scheduled task
S.DR = [];      % Dropped task
S.ND = [];      % Not dominated Set

% NodeStats contains relevant node statistics resulting from B&B search
NodeStats = struct('T',[],'PF',(1:N)','NS',[],'DR',[],'ND',[], ...
    't_ex',zeros(N,1),'ChannelAvailableTime',zeros(K,1),'ChannelAssignment',zeros(N,1),'x',zeros(N,1),...
    'CompleteSolutionFlag',0,'BestCost',Inf,'BestSeq',[]);
% NodeStats = struct('T',[],'Tterminal',[],'NS',[],'ND',[],'CompleteSolutionFlag',0,'Cost',Inf);
NodeCnt = 1; % Counter of number recorded nodes


% Intialize Channel Availability Tracker
% ChannelAvailableTimeInput = zeros(K,1);

% new_node = 1;
while ~isempty(S)
    
    % Pull off data from end of stack
    %     UB = S(end).UB;
    PF = S(end).PF;
    T = S(end).T;
    NS = S(end).NS;
    DR = S(end).DR;
    ND = S(end).ND;
    TimeExecutionInput = S(end).t_ex;
    ChannelAvailableTimeInput = S(end).ChannelAvailableTime;
    ChannelAssignmentInput = S(end).ChannelAssignment;
    ScheduledIndicatorInput = S(end).x; % x indicates whether channel has been scheduled (1) or not scheduled (0)
    
    %     [t_ex,x,ChannelAvailableTimeProposed,TaskChannel] = BBMultiChannelSequenceScheduler(T,s_task,deadline_task,length_task,ChannelAvailableTimeInput);
    %     Cprime = sum( w_task(T).*(t_ex(T) - s_task(T)) )  + sum( drop_task(DR) ) ;
    %     Cupper = sum( w_task(T).*(t_ex(T) - s_task(T)) )  + sum( drop_task(PF) ) ;
    
    
    
    %     if Cprime < UB
    
    
    if ~isempty(PF)
        
        [~,task_index] = min(s_task(PF));
        
        if all( s_task(task_index) == s_task) % All of the start times are the same. Sort by weights
            [~,task_index] = max(w_task(PF));
        end
        
        
        
        %             if new_node
        %                 new_node = 0;
        %             else
        %                 task_index = randi(length(PF),1); % Randomly choose a task to schedule
        %             end
        task = PF(task_index);
        PF(task_index) = [];
        Tprime = [T; task];
        PFprime  = [PF; NS];
        NSprime = [];
        DRprime = DR;
        NDprime = [];
        NS = [NS; task];
        
        % Update S(end)
        S(end).PF = PF;
        S(end).NS = NS;
        
        % Update Schedule Parameters After adding latest task
        %             x_input = (t_ex < deadline_task);
        [t_ex,x,ChannelAvailableTimeProposed,TaskChannel] = BBUpdateSequenceParameters(Tprime,s_task,deadline_task,length_task,ChannelAvailableTimeInput,TimeExecutionInput,ChannelAssignmentInput,ScheduledIndicatorInput);
%         drop_index = find(x(Tprime) == 0);
%         if ~isempty(drop_index) % Check to see if proposed task is dropped
%             t_ex(Tprime(drop_index)) = 0;
%             DRprime = [DRprime; Tprime(drop_index)]; % Add to drop list
%             Tprime(drop_index) = [];
%             keyboard
%         end
        
        %             [t_ex,x,ChannelAvailableTimeProposed,TaskChannel] = BBMultiChannelSequenceScheduler(Tprime,s_task,deadline_task,length_task,ChannelAvailableTimeInput);
        
        
        %                 [~,t_ex,~] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
        ExecutionTimeOffsetFlag = diff([0; t_ex(Tprime)]) >= 0 ;
        if all(ExecutionTimeOffsetFlag) % Start Times domainance rule
            
            % Move any task whose deadline has passed on all
            % timelines from PFprime to DRprime
            deadline_indicator = zeros(N,K);
            for kk = 1:K
                deadline_indicator(:,kk) =   deadline_task <  ChannelAvailableTimeProposed(kk);
            end
            pass_index = find( sum(deadline_indicator(PFprime,:),2) == K );
            if ~isempty(pass_index) % Move tasks from PFprime to DRprime
                DRprime = [DRprime; PFprime(pass_index)];
                PFprime(pass_index) = [];
                %                     keyboard
            end
            % Compute Partial Schedule Cost
            Ttemp = [Tprime; DRprime];
            [Cprime] = PartialMultiChannelSequenceScheduler(Ttemp,N,K,s_task,w_task,deadline_task,length_task,drop_task);

%             Cprime = sum( w_task(Tprime).*(t_ex(Tprime) - s_task(Tprime)) )  + sum( drop_task(DRprime) ) ;
            
            % Check if schedule is active (PFprime checked to see
            % if any can task be scheduled right before last task
            % in Tprime on same timeline without imposing a delay
            active_flag = 1;
            PartialSchedule = Tprime(1:end-1);
            ChannelSelected = TaskChannel(Tprime(end));
            %                 [t_partial,x_partial,ChannelAvailableTimePartial,PartialChannelAssignment] = BBMultiChannelSequenceScheduler(PartialSchedule,s_task,deadline_task,length_task,zeros(K,1));
            %                 [t_partial,x_partial,ChannelAvailableTimePartial,PartialChannelAssignment] = BBActiveSequenceScheduler(PartialSchedule,s_task,deadline_task,length_task,zeros(K,1),ChannelSelected);
            
            
            
            t_partial = TimeExecutionInput;
            ChannelAvailableTimePartial = ChannelAvailableTimeInput;
            PartialChannelAssignment = ChannelAssignmentInput;
            x_partial = ScheduledIndicatorInput;
            
            if sum(ChannelAvailableTimeInput - ChannelAvailableTimePartial) ~= 0
                keyboard
            end
            
            
            
            for jj = 1:length(PFprime)
                curTask = PFprime(jj);
                Tactive = [PartialSchedule; curTask; Tprime(end)];
                %                     ChannelSelected = TaskChannel(Tprime(end));
                
                [t_active] = BBPartialSequenceScheduler(t_partial,x_partial,Tactive,s_task,deadline_task,length_task,ChannelAvailableTimePartial,PartialChannelAssignment,ChannelSelected);               
                
                %                     [t_active] = BBActiveSequenceScheduler(Tactive,s_task,deadline_task,length_task,ChannelAvailableTimeInput,ChannelSelected);
                if t_active(Tprime(end)) <= t_ex(Tprime(end))   %( s_task(curTask) + length_task(curTask) ) < t_ex(Tprime(end))
                    active_flag = 0; %
                    %                     keyboard
                    break
                end
            end
            
            
            
            
            if active_flag
                
                if isempty(PFprime)
                    S(end).CompleteSolutionFlag = 1;
                    if Cprime < S(end).BestCost
                        S(end).BestCost = Cprime;
                        S(end).BestSeq = [Tprime; DRprime];
                        if ~exist('SeqBest','var')
                            SeqBest = [Tprime; DRprime];
                            [Cbest] = MultiChannelSequenceScheduler(SeqBest,N,K,s_task,w_task,deadline_task,length_task,drop_task);
                        end
                    end
                else
%                     keyboard

                    CurSeqBest = S(end).BestSeq;                    
                    SeqBest = [Tprime; PFprime; DRprime];
                    
                    if isempty(CurSeqBest) 
                        CurSeqBest = zeros(N,1);                        
                    end                    
                    deltaSeq = SeqBest - CurSeqBest;
                    
                    if any(deltaSeq)
                        
                        if length(SeqBest) ~= N
                            keyboard
                        end
                        [Cbest] = MultiChannelSequenceScheduler(SeqBest,N,K,s_task,w_task,deadline_task,length_task,drop_task);
                        if Cbest < S(end).BestCost
                            S(end).BestCost = Cbest;
                            S(end).BestSeq = SeqBest;
                        end
                        % Assume All tasks of PFprime are dropped and update
                        % terminal cost accordingly (need to do this because
                        % lows-active flag is not being used
                    else
                        Cbest = S(end).BestCost;                        
                    end
                    
                end
                
                if Cprime < UB
                    
                    % Update ND set
                    ND = ([ND; task]);
                    %                     NodeStats(end).ND = ND; % Update node statistics
%                     if length(ND) > 1
%                         keyboard
%                     end
                    
                    S(end).ND = ND;
                    
                    %                         LowsFlag = LOWS_active(K,Tprime,PFprime,DRprime,TaskChannel,s_task,w_task,deadline_task,drop_task,length_task,ChannelAvailableTimeInput);
                    %                         if LowsFlag  % Not implemented correctly. Just
                    %                         brute force w/o checking this step.
                    
                    % Push (Tprime,PFprime,NSprime,DRprime) onto
                    % stack
                    %                     Snew.UB = inf;
                    Snew.T = Tprime;
                    Snew.PF = PFprime;
                    Snew.NS = NSprime;
                    Snew.DR = DRprime;
                    Snew.ND = NDprime;
                    Snew.t_ex = t_ex;
                    Snew.x = x;
                    Snew.ChannelAvailableTime = ChannelAvailableTimeProposed;
                    Snew.ChannelAssignment = TaskChannel;
                    Snew.BestSeq = SeqBest;
                    Snew.BestCost = Cbest;
                    
                    
                    S = [S; Snew];
                    %                             new_node = 1;
                    %                         end
                end
                
            end
            
            
        end
    else
        
        %             keyboard
        
        
        
        
%         [t_ex,x,ChannelAvailableTimeProposed,TaskChannel] = BBMultiChannelSequenceScheduler(T,s_task,deadline_task,length_task,zeros(K,1));
        
        t_ex = TimeExecutionInput;
        
%         if ~all(TimeExecutionInput == t_ex)
%             keyboard
%         end
            
        
%         C = sum( w_task(T).*(t_ex(T) - s_task(T)) )  + sum( drop_task(DR) ) ;
        Ttemp = [T; NS; DR;]; % Place the not scheduled tasks on timeline, cause that is what really happens
        [C] = PartialMultiChannelSequenceScheduler(Ttemp,N,K,s_task,w_task,deadline_task,length_task,drop_task);
%         C = MultiChannelSequenceScheduler(Ttemp,N,K,s_task,w_task,deadline_task,length_task,drop_task);
%         if C2 ~= C
%             keyboard
%         end
        
        if isempty(NS) && C < UB
            UB = C;
            Tstar = T;
            Tdr = DR;
            Tfinal = [T; DR];
        end
        if S(end).CompleteSolutionFlag == 1
            %             S(end).CompleteSolutionFlag = 1;
            if C < S(end).BestCost && length(T) == N
                S(end).BestCost = C;
                S(end).BestSeq = T;
                keyboard
            end
            NodeStats(NodeCnt) = S(end);
            NodeCnt = NodeCnt + 1;
            %             keyboard
        end
        S(end) = [];
        
        
    end
    
    %     else
    %         S(end) = [];
    %     end
    
end


fields = {'t_ex','ChannelAvailableTime','ChannelAssignment','x'};
NodeStats = rmfield(NodeStats,fields);