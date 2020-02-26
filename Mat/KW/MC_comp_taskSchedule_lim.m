%%%%%%%%%% Task Scheduling Comparison

clear;

% rng(107);


%%% Inputs
% t_run_max = (0:0.02:0.2)';
t_run_max = (0:0.1:2.5)';


N_mc = 100;

% Algorithms
% fcn_search = {@(s_task,d_task,l_task,t_run_max) fcn_BB_lim(s_task,d_task,l_task,t_run_max,'LIFO');
%     @(s_task,d_task,l_task,t_run_max) fcn_MCTS_lim(s_task,d_task,l_task,t_run_max,100)};
fcn_search = {@(s_task,d_task,l_task,t_run_max) fcn_ES(s_task,d_task,l_task,t_run_max);
    @(s_task,d_task,l_task,t_run_max) fcn_BB_lim(s_task,d_task,l_task,t_run_max,'LIFO');
    @(s_task,d_task,l_task,t_run_max) fcn_MCTS_lim(s_task,d_task,l_task,t_run_max,100)};


% % Tasks
% N = 8;                      % number of tasks
% 
% s_task = 30*rand(N,1);            % task start times
% d_task = 1 + 2*rand(N,1);          % task durations
% 
% 
% w = 0.8 + 0.4*rand(N,1);
% t_drop = s_task + d_task.*(3+2*rand(N,1));
% l_drop = (2+rand(N,1)).*w.*(t_drop-s_task);
% 
% l_task = cell(N,1);
% for n = 1:N
%     l_task{n} = @(t) cost_linDrop(t,w(n),s_task(n),t_drop(n),l_drop(n));
% end





%%% Monte Carlo Simulation
N_t_run = numel(t_run_max);
N_alg = numel(fcn_search);

loss_mc = zeros(N_mc,N_t_run,N_alg);

for i_mc = 1:N_mc
    
    fprintf('MC %i/%i \n',i_mc,N_mc);
    
    % Tasks
    N = 8;                      % number of tasks

    s_task = 5*rand(N,1);            % task start times
    d_task = 1 + 2*rand(N,1);          % task durations


    w = 0.8 + 0.4*rand(N,1);
    t_drop = s_task + d_task.*(3+2*rand(N,1));
    l_drop = (2+rand(N,1)).*w.*(t_drop-s_task);

    l_task = cell(N,1);
    for n = 1:N
        l_task{n} = @(t) cost_linDrop(t,w(n),s_task(n),t_drop(n),l_drop(n));
    end


    for i_t = 1:N_t_run    
        
        fprintf('   Max Time %i/%i \n',i_t,N_t_run);

        for i_a = 1:N_alg
            [t_ex_min,l_min] = fcn_search{i_a}(s_task,d_task,l_task,t_run_max(i_t));
            loss_mc(i_mc,i_t,i_a) = l_min;
        end

    end
end

loss = squeeze(mean(loss_mc,1));



%%% Results

% Plots

figure(1); clf; 
plot(t_run_max,loss,'-o');
grid on; xlabel('Max Run Time'); ylabel('Avg. Loss'); %legend('BB','MCTS');
title('Scheduler Performace for Random Task Sets');

str_leg = cell(size(fcn_search));
for i_a = 1:numel(fcn_search)
    temp = func2str(fcn_search{i_a});
    i_p_1 = find(temp == '(');
    i_p_2 = find(temp == ')');
    str_leg{i_a} = temp(i_p_2(1)+5:i_p_1(2)-1);
end
legend(str_leg,'Interpreter','none');






