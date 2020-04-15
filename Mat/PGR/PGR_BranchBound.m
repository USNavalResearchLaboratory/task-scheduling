%%%%%%%%%% Branch and Bound

clear;

rng(100);


%%% Inputs

% Algorithm
mode_stack = 'LIFO';
% mode_stack = 'FIFO';


% Tasks
N = 10;                      % number of tasks

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

% Initialize Stack
LB = 0;
UB = 0;
temp = sum(d_task) + max(s_task);
for n = 1:N
    LB = LB + l_task{n}(s_task(n));
    UB = UB + l_task{n}(temp);
end

S = struct('seq',[],'t_ex',Inf(N,1),'l_inc',0,'LB',LB,'UB',UB);      % Branch Stack


% Iterate
while (numel(S) ~= 1) || (numel(S(1).seq) ~= N)
       
    fprintf('# Remaining Branches = %i \n',numel(S));
    
    % Extract Branch       
    for i = 1:numel(S)
        if numel(S(i).seq) ~= N
            B = S(i);
            S(i) = [];
            break
        end
    end
        
           
    % Split Branch
    T_c = setdiff((1:N)',B.seq);
%     seq_rem = T_c;
    seq_rem = T_c(randperm(numel(T_c)));        %%%
    for n = seq_rem' 
                
        % Generate New Branch
        B_new = branch_update(B,n,l_task,s_task,d_task);
        
        % Cut Branches
        if B_new.LB >= min(cell2mat({S.UB}))
            % New Branch is Dominated
%             fprintf('Cut\n')
        else
            % Cut Dominated Branches
            S(cell2mat({S.LB}) >= B_new.UB) = [];           
        
            % Add New Branch to Stack  
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

l_opt = S.l_inc;
t_ex_opt = S.t_ex;

t_run = toc;



%%% Results

t_ex_opt
l_opt
t_run


if numel(S) ~= 1
    error('Multiple leafs...');
end

if sum(cell2mat({S.LB}) ~= cell2mat({S.UB})) ~= 0
    error('Leaf bounds do not converge.');
end


% Cost evaluation
l_calc = 0;
for n = 1:N
    l_calc = l_calc + l_task{n}(t_ex_opt(n));
end

if abs(l_calc - l_opt) > 1e-12
    error('Iterated loss is inaccurate');
end


% Check solution validity
valid = 1;
for n_1 = 1:N-1
    for n_2 = n_1+1:N
        cond_1 = t_ex_opt(n_1) >= (t_ex_opt(n_2) + d_task(n_2));
        cond_2 = t_ex_opt(n_2) >= (t_ex_opt(n_1) + d_task(n_1));
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
    area([t_ex_opt(n); t_ex_opt(n) + d_task(n)],[1;1],0);
end
grid on; legend; title(sprintf('Optimal Schedule, Loss = %f',l_opt));
xlabel('Time'); set(gca,'YLim',[0,1.5])








