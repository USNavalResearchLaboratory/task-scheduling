function [t_ex_opt,l_opt,t_run,X,Y] = fcn_BB_NN(s_task,d_task,l_task,mode_stack,varargin)
%%%%%%%%%% Branch and Bound with Neural Network Inputs stored

if isempty(varargin)
    curTime = 0;
else
    curTime = varargin{1};
end

tic;

N = numel(l_task);

% Initialize Stack
LB = 0;
UB = 0;
temp = sum(d_task) + max(s_task);
for n = 1:N
    LB = LB + l_task{n}(s_task(n));
    UB = UB + l_task{n}(temp);    
    [~,w(n),t_start(n),t_drop(n),c_drop(n)] = l_task{n}(0);
    
end

% Assign Static Policy Features (PF)
PF(1,:) = s_task; % Start time of task
PF(2,:) = d_task; % Task Duration
PF(3,:) = w; % Task Slope
PF(4,:) = t_drop; % Task Drop Time
PF(5,:) = c_drop; % Task Drop Cost
PF_node = zeros(N,N);
Label = ones(N,N);




S = struct('seq',[],'t_ex',Inf(N,1),'l_inc',0,'LB',LB,'UB',UB,'Tc',[1:N]','DOM',0);      % Branch Stack
S_term = struct('seq',[],'t_ex',Inf(N,1),'l_inc',0,'LB',LB,'UB',UB,'Tc',[1:N]','DOM',0);      % Branch Stack

MAX_CNT = 100000;
S_all(1:MAX_CNT,1) = struct('seq',[],'t_ex',Inf(N,1),'l_inc',0,'LB',LB,'UB',UB,'Tc',[1:N]','DOM',0);      % Branch Stack
B_new = struct('seq',[],'t_ex',Inf(N,1),'l_inc',0,'LB',LB,'UB',UB,'Tc',[1:N]','DOM',0);      % Branch Stack
S_LB = 1;
S_UB = 1;

% Iterate
node_cnt = 1;
cnt = 1;
while (numel(S) ~= 1) || (numel(S(1).seq) ~= N)
       
%     fprintf('# Remaining Branches = %i \n',numel(S));
    
    % Extract Branch       
    for i = 1:numel(S)
        if numel(S(i).seq) ~= N
            B = S(i);
            S(i) = [];
            S_LB(i) = [];
            S_UB(i) = [];
            break
        end
    end
    
%     S_all(node_cnt) = B;
%     DOM_Task = [];
        
           
    % Split Branch
%     T_c = setdiff((1:N)',B.seq);
    T_c = B.Tc;
    seq_rem = T_c(randperm(numel(T_c)));        %%%
    for n = seq_rem' 
                
        % Generate New Branch
        B_new = branch_update(B,n,l_task,s_task,d_task,w,t_drop,c_drop,B_new,curTime);
        S_all(cnt) = B_new;
        cnt = cnt + 1;
%         S_all = [B_new; S_all];
        
%         label = zeros(N,1);
%         label(B_new.seq(end)) = 1; % One hot encode branches
        
        % Cut Branches
%         if B_new.LB >= min(cell2mat({S.UB}))
        if B_new.LB >= min(S_UB)
            % New Branch is Dominated
%             DOM_Task = [DOM_Task; n];
%             keyboard
%             B_new.DOM = 1;
%             S_all = [B_new; S_all];

            
% %             PF_node = encode_sequence(B.seq,N);
%             task_idx = B_new.seq(end);
%             time_idx = length(B_new.seq);
%             Label(time_idx,task_idx) = 0;
            
%             PF_status( = 
            
%             STATUS(:,B_new.seq(end)) =  [0; 1; 0 ]; % Dominated indicator
        else
            % Cut Dominated Branches
            DOM_INDEX = find( S_LB >= B_new.UB);
            if ~isempty(DOM_INDEX)
%                 keyboard
%                 for jj = 1:length(DOM_INDEX)
%                     task_idx = S(DOM_INDEX(jj)).seq(end);
%                     time_idx = length(S(DOM_INDEX(jj)).seq);
%                     Label(time_idx,task_idx) = 0;
%                 end
%                 for kk = find(cell2mat({S_all.LB}) >= B_new.UB)
%                     S_all(kk).DOM = 1;
%                 end               

%                 S_all = [B_new; S_all];
            else
%                 S_all = [B_new; S_all];

            end
            
%             S_LB  >= B_new.UB
            
            S( S_LB  >= B_new.UB) = [];
            S_LB( S_LB  >= B_new.UB )  = [];
            S_UB( S_LB  >= B_new.UB ) = [];
            
            
            % Add New Branch to Stack  
            if strcmpi(mode_stack,'FIFO')
                S = [S; B_new];
            elseif strcmpi(mode_stack,'LIFO')
                S = [B_new; S];
                S_LB = [B_new.LB; S_LB];
                S_UB = [B_new.UB; S_UB];
            else
                error('Unsupported stacking function.');
            end
            
            if length(B_new.seq) == N
                S_term = [B_new; S_term];
%                 keyboard
            end

        end

                        
    end
%     S_all(node_cnt).DOM = DOM_Task;
%     node_cnt = node_cnt + 1;
            
end
S_term(end) = [];
S_all(cnt:end) = [];
% S_all(end) = [];

plot_en = 0;
seq_opt = S.seq; % Optimal Sequence
if length(S_all) > 10000
%     keyboard
%     S_all(1:end-10000) = [];    
end

node_stats = generate_node_statistics_FAST(S_all,N,seq_opt,plot_en);
% node_stats2 = generate_node_statistics(S_all,N,seq_opt,plot_en);


l_opt = S.l_inc;
t_ex_opt = S.t_ex;

t_run = toc;

[X,Y] = create_train_samples(PF,node_stats,N);

%% Generate Input data to train policy network
% policy_input(1,:) = s_task; % Start time of task
% policy_input(2,:) = d_task; % Task Duration




