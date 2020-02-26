function [t_ex_min,l_min] = fcn_BB_lim(s_task,d_task,l_task,t_run_max,mode_stack)
%%%%%%%%%% Branch and Bound

tic;

N = numel(l_task);

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
       
%     fprintf('# Remaining Branches = %i \n',numel(S));
    
    % Extract Branch    
    B = S(1);
    S(1) = [];
       
           
    % Split Branch
    T_c = setdiff((1:N)',B.seq);
    seq_rem = T_c(randperm(numel(T_c)));        %%% random???
    for n = seq_rem' 
                
        % Check Runtime
        t_run = toc;
        if t_run >= t_run_max
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






