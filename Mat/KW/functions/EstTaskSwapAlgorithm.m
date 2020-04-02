function [Cost,t_ex,NumDropTask,T,ChannelAvailableTime] = EstTaskSwapAlgorithm(data)

% Earliest Start Time Algorithm
% Takes tasks in data and assigns them to timeline using the EST algorithm.
% If any tasks are dropped performs swapping between adjacent tasks to find
% better timeline

% Input: data summarized below

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



[~,T] = sort(s_task); % Sort jobs based on starting times

% Assign those tasks to a timeline.
if ~strcmpi(data.scheduler,'flexdar')
    [Cost,t_ex,NumDropTask] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
%         [T,Cost.EST(monte,cnt.N),t_ex,NumDropTask] = EST_MultiChannel(N,K,s_task,w_task,deadline_task,length_task,drop_task);
else
    [Cost,t_ex,ChannelAvailableTime,NumDropTask] = FlexDARMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task,ChannelAvailableTime,RP);    
end

% Perform Task Swapping for EST
if NumDropTask > 0
    for jj = 1:N-1
        Tswap = T;
        T1 = T(jj);
        T2 = T(jj+1);
        Tswap(jj) = T2;
        Tswap(jj+1) = T1;        
        if ~strcmpi(data.scheduler,'flexdar')
            [~,t_ex] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
        else
            [~,t_ex] = FlexDARMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task,ChannelAvailableTime,RP);
        end        
%         [~,t_ex] = MultiChannelSequenceScheduler(Tswap,N,K,s_task,w_task,deadline_task,length_task,drop_task);
        if sum( t_ex < deadline_task ) == N
            T = Tswap;
            keyboard
            break
        end
    end
    if ~strcmpi(data.scheduler,'flexdar')
        [Cost,t_ex,NumDropTask] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
    else
        [Cost,t_ex,ChannelAvailableTime,NumDropTask] = FlexDARMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task,ChannelAvailableTime,RP);
    end
    
%     [Cost,t_ex,NumDropTask] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
%     DropPercent.EstSwap(monte,cnt.N) = NumDropTask/N;
%     RunTime.EstSwap(monte,cnt.N) = toc;
else
%     Cost.EstSwap(monte,cnt.N) = Cost.EST(monte,cnt.N);
%     DropPercent.EstSwap(monte,cnt.N) = DropPercent.EST(monte,cnt.N);
%     RunTime.EstSwap(monte,cnt.N) = RunTime.EST(monte,cnt.N);
end


