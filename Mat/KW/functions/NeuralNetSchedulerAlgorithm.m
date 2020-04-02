function [Cost,t_ex,NumDropTask,T,ChannelAvailableTime] = NeuralNetSchedulerAlgorithm(data,net)

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

PF(1,:) = s_task;
PF(2,:) = deadline_task;
PF(3,:) = length_task;
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

T = node;


if ~strcmpi(data.scheduler,'flexdar')    
    [Cost,t_ex,NumDropTask] = MultiChannelSequenceScheduler(node,N,K,s_task,w_task,deadline_task,length_task,drop_task);
else
    [Cost,t_ex,ChannelAvailableTime,NumDropTask] = FlexDARMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task,ChannelAvailableTime,RP);
end


