%%%%%%%%%% Monte Carlo Tree Search

clear;

rng(107);


%%% Inputs

% Algorithm
N_mc = 10000;


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





%%% Tree Search

tic;

t_ex = Inf(N,1);
seq = [];
l_inc = 0;

l_min = Inf;

for i_t = 1:N
    
    fprintf('Assigning Task %i/%i ... \n',i_t,N);
       
    % Perform Rollouts
    T_c = setdiff((1:N)',seq);

    for i_mc = 1:N_mc
        
        % Random sequence
        seq_rem = T_c(randperm(numel(T_c)));       
              
        % Determine execution times, Assess loss       
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



%%% Results

t_ex_min
l_min
time_elapsed


% Cost evaluation
l_calc = 0;
for n = 1:N
    l_calc = l_calc + l_task{n}(t_ex_min(n));
end

if abs(l_calc - l_min) > 1e-14
    error('Iterated loss is inaccurate');
end


% Check solution validity
valid = 1;
for n_1 = 1:N-1
    for n_2 = n_1+1:N
        cond_1 = t_ex_min(n_1) >= (t_ex_min(n_2) + d_task(n_2));
        cond_2 = t_ex_min(n_2) >= (t_ex_min(n_1) + d_task(n_1));
        valid = valid && (cond_1 || cond_2);
    end
end

if ~valid
    error('Invalid Solution: Scheduling Conflict');
end




%%% Plots

t_plot = 0:0.01:ceil(max(t_drop));
figure(1); clf;
for n = 1:N
    v = zeros(size(t_plot));
    for idx = 1:numel(t_plot)
        v(idx) = l_task{n}(t_plot(idx));
    end
    plot(t_plot,v); grid on; 
    xlabel('t'); ylabel('Loss'); title('Task Losses'); legend('Location','NorthWest'); 
    hold on;
end

figure(2); clf; hold on;
for n = 1:N
    area([t_ex_min(n); t_ex_min(n) + d_task(n)],[1;1],0);
end
grid on; legend; title(sprintf('Optimal Schedule, Loss = %f',l_min));
xlabel('Time'); set(gca,'YLim',[0,1.5])









