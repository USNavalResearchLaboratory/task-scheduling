function [t_ex_opt,l_opt,t_run] = fcn_BB(s_task,d_task,l_task,mode_stack)
%%%%%%%%%% Branch and Bound

tic;

N = numel(l_task);

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
       
%     fprintf('# Remaining Branches = %i \n',numel(S));
    
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
    seq_rem = T_c(randperm(numel(T_c)));        %%%
    for n = seq_rem' 
                
        % Generate New Branch
        B_new = branch_update(B,n,l_task,s_task,d_task);
                     
        
        % Cut Branches
        if B_new.LB >= min(cell2mat({S.UB}))
            % New Branch is Dominated
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







