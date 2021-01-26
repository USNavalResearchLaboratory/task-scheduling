function [Cost,t_ex,NumDropTask,T,ChannelAvailableTime] = BbQueueAlgorithm(data)

% Earliest Deadline Algorithm
% Takes tasks in data and assigns them to timeline using the Earliest deadline.
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

try
[T,~,~] = BBschedulerQueueVersion(K,s_task,deadline_task,length_task,drop_task,w_task,ChannelAvailableTime);
catch
    keyboard
end
    

if ~strcmpi(data.scheduler,'flexdar')    
    [Cost,t_ex,NumDropTask] = FunctionMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task,ChannelAvailableTime);
else
    [Cost,t_ex,ChannelAvailableTime,NumDropTask] = FlexDARMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task,ChannelAvailableTime,RP);
end


   

