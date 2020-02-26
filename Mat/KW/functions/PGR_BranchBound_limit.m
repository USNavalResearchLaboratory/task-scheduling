%%%%%%%%%% Branch and Bound

clear;

% rng(300);


%%% Inputs

t_run_max = Inf;                  % max algorithm runtime (s)

% Algorithm
mode_stack = 'LIFO';
% mode_stack = 'FIFO';


% Tasks
N = 8;                      % number of tasks

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

% Initial Rollout
S_f = struct('seq',[],'t_ex',Inf(N,1),'l',0);

T_c = (1:N)';
for n = T_c(randperm(N))'    
    if isempty(S_f.seq)
        S_f.t_ex(n) = s_task(n);
    else
        S_f.t_ex(n) = max([s_task(n); S_f.t_ex(S_f.seq(end)) + d_task(S_f.seq(end))]);
    end

    S_f.seq = [S_f.seq; n];
    S_f.l = S_f.l + l_task{n}(S_f.t_ex(n));
end




% Initialize Stack
LB = 0;
UB = 0;
temp = sum(d_task) + max(s_task);
for n = 1:N
    LB = LB + l_task{n}(s_task(n));
    UB = UB + l_task{n}(temp);
end

S = struct('seq',[],'t_ex',Inf(N,1),'l_inc',0,'LB',LB,'UB',UB);      % Branch Stack


% Search
flag_run = 1;
while ~isempty(S) && flag_run
       
    fprintf('# Remaining Branches = %i \n',numel(S));
    
    % Extract Branch    
    B = S(1);
    S(1) = [];
       
           
    % Split Branch
    T_c = setdiff((1:N)',B.seq);
    seq_rem = T_c(randperm(numel(T_c)));        %%% random???
    for n = seq_rem' 
                
        % Check Runtime
        t_run = toc;
        if toc >= t_run_max
            flag_run = 0;
            break
        end
        
        
        % Generate New Branch
        B_new = branch_update(B,n,l_task,s_task,d_task);
                     
        
        % Check if New Branch is Dominated       
        if B_new.LB < min([cell2mat({S.UB}), S_f.l]) 
            
            % Cut Any Dominated Branches
            S(B_new.UB <= cell2mat({S.LB})) = [];
               
            
            % Rollout/Reassign if Solution is Dominated
            if B_new.UB < S_f.l
                
                S_f = struct('seq',B_new.seq,'t_ex',B_new.t_ex,'l',B_new.l_inc);

                T_c = setdiff((1:N)',B_new.seq);                    
                for n_mc = T_c(randperm(numel(T_c)))'    
                    if isempty(S_f.seq)
                        S_f.t_ex(n_mc) = s_task(n_mc);
                    else
                        S_f.t_ex(n_mc) = max([s_task(n_mc); S_f.t_ex(S_f.seq(end)) + d_task(S_f.seq(end))]);
                    end

                    S_f.seq = [S_f.seq; n_mc];
                    S_f.l = S_f.l + l_task{n_mc}(S_f.t_ex(n_mc));
                end

            end
            
            
            % Return Partial Sequence to Stack  
            if numel(B_new.seq) ~= N   % complete sequence                             
                if strcmpi(mode_stack,'FIFO')
                    S = [S; B_new];
                elseif strcmpi(mode_stack,'LIFO')
                    S = [B_new; S];
                else
                    error('Unsupported stacking function.');
                end
            end
            
            
        end
                                                       
    end
    
end

l_min = S_f.l;
t_ex_min = S_f.t_ex;




%%% Results

t_ex_min
l_min
t_run

if flag_run
    fprintf('Optimal Solution Found. \n');
else
    fprintf('Timelimit Reached. \n');
end



% Cost evaluation
l_calc = 0;
for n = 1:N
    l_calc = l_calc + l_task{n}(t_ex_min(n));
end

if abs(l_calc - l_min) > 1e-12
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




% %%% Plots
% 
% t_plot = 0:0.01:ceil(max(t_drop));
% figure(1); clf;
% for n = 1:N
%     v = zeros(size(t_plot));
%     for idx = 1:numel(t_plot)
%         v(idx) = l_task{n}(t_plot(idx));
%     end
%     plot(t_plot,v); grid on; 
%     xlabel('t'); ylabel('Loss'); title('Task Losses'); legend('Location','NorthWest'); 
%     hold on;
% end
% 
% figure(2); clf; hold on;
% for n = 1:N
%     area([t_ex_min(n); t_ex_min(n) + d_task(n)],[1;1],0);
% end
% grid on; legend; title(sprintf('Optimal Schedule, Loss = %f',l_min));
% xlabel('Time'); set(gca,'YLim',[0,1.5])
% 







