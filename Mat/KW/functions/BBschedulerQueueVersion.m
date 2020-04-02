function [Tfinal,Tstar,Tdr] = BBschedulerQueueVersion(K,s_task,deadline_task,length_task,drop_task,w_task,curTime)


N = length(s_task);
% Initialize Stack

UB = inf; % Upper Bound
S = struct('T',[],'PF',(1:N)','NS',[],'DR',[],'t_ex',zeros(N,1),'ChannelAvailableTime',curTime,'ChannelAssignment',zeros(N,1),'x',zeros(N,1));
Snew = S;


S.T = [];   % Task sequence
S.PF = (1:N)'; % Possible First Task
S.NS = [];      % Not scheduled task
S.DR = [];      % Dropped task


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
        NS = [NS; task];
        
        % Update S(end)
        S(end).PF = PF;
        S(end).NS = NS;
        
        % Update Schedule Parameters After adding latest task
        %             x_input = (t_ex < deadline_task);
        [t_ex,x,ChannelAvailableTimeProposed,TaskChannel] = BBUpdateSequenceParameters(Tprime,s_task,deadline_task,length_task,ChannelAvailableTimeInput,TimeExecutionInput,ChannelAssignmentInput,ScheduledIndicatorInput);
        
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
            Cprime = sum( w_task(Tprime).*(t_ex(Tprime) - s_task(Tprime)) )  + sum( drop_task(DRprime) ) ;
            
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
                
                try
                    [t_active] = BBPartialSequenceScheduler(t_partial,x_partial,Tactive,s_task,deadline_task,length_task,ChannelAvailableTimePartial,PartialChannelAssignment,ChannelSelected);
                catch
                    keyboard
                end
                
                %                     [t_active] = BBActiveSequenceScheduler(Tactive,s_task,deadline_task,length_task,ChannelAvailableTimeInput,ChannelSelected);
                
                
                if t_active(Tprime(end)) <= t_ex(Tprime(end))   %( s_task(curTask) + length_task(curTask) ) < t_ex(Tprime(end))
                    active_flag = 0; %
                    %                     keyboard
                    break
                end
            end
            
            
            
            
            if active_flag
                if Cprime < UB
                    
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
                    Snew.t_ex = t_ex;
                    Snew.x = x;
                    Snew.ChannelAvailableTime = ChannelAvailableTimeProposed;
                    Snew.ChannelAssignment = TaskChannel;
                    
                    
                    S = [S; Snew];
                    %                             new_node = 1;
                    %                         end
                end
                
            end
            
            
        end
    else
        
        %             keyboard
        
        
        [t_ex,x,ChannelAvailableTimeProposed,TaskChannel] = BBMultiChannelSequenceScheduler(T,s_task,deadline_task,length_task,curTime);
        
        C = sum( w_task(T).*(t_ex(T) - s_task(T)) )  + sum( drop_task(DR) ) ;
        if isempty(NS) && C < UB
            UB = C;
            Tstar = T;
            Tdr = DR;
            Tfinal = [T; DR];
        end
        S(end) = [];
        
        
    end
    
    %     else
    %         S(end) = [];
    %     end
    
end