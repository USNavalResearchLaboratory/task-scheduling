%%%%%%%%%% Task Scheduling Comparison

clear;

% rng(107);


%%% Inputs
N_mc = 10;

% Algorithms
% fcn_search = {@(s_task,d_task,l_task) fcn_BB(s_task,d_task,l_task,'LIFO');
%     @(s_task,d_task,l_task) fcn_MCTS_fast(s_task,d_task,l_task,1000)};
fcn_search = {@(s_task,d_task,l_task) fcn_BB(s_task,d_task,l_task,'LIFO');
    @(s_task,d_task,l_task) fcn_MCTS(s_task,d_task,l_task,1000)};


% Tasks
N = 12;                      % number of tasks

s_task = 30*rand(N,1);            % task start times
d_task = 1 + 2*rand(N,1);          % task durations


w = 0.8 + 0.4*rand(N,1);
t_drop = s_task + d_task.*(3+2*rand(N,1));
l_drop = (2+rand(N,1)).*w.*(t_drop-s_task);

l_task = cell(N,1);
for n = 1:N
    l_task{n} = @(t) cost_linDrop(t,w(n),s_task(n),t_drop(n),l_drop(n));
end





%%% Monte Carlo Simulation
N_alg = numel(fcn_search);

loss_mc = zeros(N_mc,N_alg);
t_run_mc = zeros(N_mc,N_alg);
for i_mc = 1:N_mc
        
    fprintf('Task Set %i/%i \n',i_mc,N_mc);
    
%     % Tasks
%     N = 8;                      % number of tasks
% 
%     s_task = 30*rand(N,1);            % task start times
%     d_task = 1 + 2*rand(N,1);          % task durations
% 
% 
%     w = 0.8 + 0.4*rand(N,1);
%     t_drop = s_task + d_task.*(3+2*rand(N,1));
%     l_drop = (2+rand(N,1)).*w.*(t_drop-s_task);
% 
%     l_task = cell(N,1);
%     for n = 1:N
%         l_task{n} = @(t) cost_linDrop(t,w(n),s_task(n),t_drop(n),l_drop(n));
%     end


    % Search
    for i_a = 1:N_alg
        [t_ex,loss,t_run] = fcn_search{i_a}(s_task,d_task,l_task);
        
        loss_mc(i_mc,i_a) = loss;
        t_run_mc(i_mc,i_a) = t_run;      
    end

end




%%% Results

% Plots

figure(1); clf; 
plot(t_run_mc(:,1),loss_mc(:,1),'b.',t_run_mc(:,2),loss_mc(:,2),'r.');
grid on; xlabel('Run Time'); ylabel('Loss'); legend('BB','MCTS');
title('Scheduler Performace for Random Task Sets');
