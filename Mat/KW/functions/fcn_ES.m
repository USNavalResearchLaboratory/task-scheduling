function [t_ex_min,l_min] = fcn_ES(s_task,d_task,l_task,t_run_max)
%%%%%%%%%% Earliest Start Time

N = numel(l_task);

[~,seq] = sort(s_task);


t_ex_min = Inf(N,1);
l_min = 0;

n = seq(1);
t_ex_min(n) = s_task(n);
l_min = l_min + l_task{n}(t_ex_min(n));

for i_t = 2:N 
    
    n = seq(i_t);
    n_prev = seq(i_t-1);
    
    t_ex_min(n) = max([s_task(n); t_ex_min(n_prev) + d_task(n_prev)]);
    l_min = l_min + l_task{n}(t_ex_min(n));
end

