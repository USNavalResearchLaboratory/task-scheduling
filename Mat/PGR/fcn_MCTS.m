function [t_ex_min,l_min,time_elapsed] = fcn_MCTS(s_task,d_task,l_task,N_mc)
%%%%%%%%%% Monte Carlo Tree Search

tic;

N = numel(l_task);

seq = [];
t_ex = Inf(N,1);
l_inc = 0;

l_min = Inf;

for i_t = 1:N
    
%     fprintf('Assigning Task %i/%i ... \n',i_t,N);
       
    % Perform Rollouts
    T_c = setdiff((1:N)',seq);

    for i_mc = 1:N_mc
        
        % Random sequence
        seq_rem = T_c(randperm(numel(T_c)));       
        
        
        % Assess loss       
        t_ex_mc = t_ex;
        seq_mc = seq;
        l_mc = l_inc;
        for n = seq_rem'    
            if isempty(seq_mc)
                t_ex_mc(n) = s_task(n);
            else
                t_ex_mc(n) = max([s_task(n); t_ex_mc(seq_mc(end)) + d_task(seq_mc(end))]);
            end
            
            seq_mc = [seq_mc; n];
            l_mc = l_mc + l_task{n}(t_ex_mc(n));
        end

        % Update best branch
        if l_mc < l_min
            t_ex_min = t_ex_mc;
            seq_min = seq_mc;
            l_min = l_mc;
        end

    end
    
    
    % Update sequence 
    t_ex(seq_min(i_t)) = t_ex_min(seq_min(i_t));

    seq = [seq; seq_min(i_t)];
    l_inc = l_inc + l_task{seq_min(i_t)}(t_ex(seq_min(i_t)));
    
    
end

time_elapsed = toc;












