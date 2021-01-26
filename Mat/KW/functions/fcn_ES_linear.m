function [t_ex_min,l_min,t_run] = fcn_ES_linear(s_task,d_task,w_task,curTime)
%%%%%%%%%% Earliest Start Time
tic;
 
N = numel(s_task);

[~,seq] = sort(s_task);


t_ex_min = Inf(N,1);
l_min = 0;

n = seq(1);
t_ex_min(n) = max( s_task(n) , curTime );
% l_min = l_min + l_task{n}(t_ex_min(n));
l_min = l_min + cost_linear(curTime,w_task(n),s_task(n));

for i_t = 2:N 
    
    n = seq(i_t);
    n_prev = seq(i_t-1);
    
    t_ex_min(n) = max([s_task(n); t_ex_min(n_prev) + d_task(n_prev)]);
    l_min = l_min + cost_linear(t_ex_min(n),w_task(n),s_task(n));    
%     l_min = l_min + l_task{n}(t_ex_min(n));
end

t_run = toc;