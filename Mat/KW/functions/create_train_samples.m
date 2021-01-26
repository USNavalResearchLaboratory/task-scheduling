function [X,Y] = create_train_samples(PF,node_stats,N)

% PF - policy feature data common to all nodes in the tree
%       includes dropping costs, cost slopes, task start times, etc...
% opt_seq - optimal sequence of actions after B&B
% N - number of tasks to assign


%% Enumerate all terminal sequence from B&B and generate associated node policy features


N_nodes = length(node_stats);

num_row = size(PF,1) + N + 3;

X = zeros(num_row , N , N_nodes);
Y = zeros(N_nodes,N);

for jj = 1:N_nodes
    
    node_seq = node_stats(jj).seq;
    try
        PF_node = encode_sequence(node_seq,N);
    catch
        keyboard
    end
    
    label = zeros(1,N);
    label(node_stats(jj).opt_a) = 1;
    
    PF_status = zeros(3,N);
    PF_status(1,:) = 1;
    for kk = 1:length(node_stats(jj).DOM)
        PF_status(:,node_stats(jj).DOM(kk)) = [0; 1; 0]; % Dominated Node
    end
    for kk = 1:length(node_seq)    
        PF_status(:,node_seq(kk)) = [0; 0; 1];  % Infeasible already assigned
    end
%     PF_feasible = ones(1,N);
%     PF_feasible(node_seq) = 0; 
    
    X(:,:,jj) = [PF; PF_status; PF_node; ];
    Y(jj,:) = label;
    
    
end



% if 0
%     %% Handle Root Node
%     % cnt = 1;
%     % root_seq = [];
%     %
%     % PF_node = encode_sequence(root_seq,N);
%     %
%     % idx = length(root_seq)+1;
%     % label = zeros(1,N);
%     % label(opt_seq(idx)) = 1;
%     %
%     %
%     % X(:,:,cnt) = [PF; PF_node];
%     % Y(cnt,:) = label;
%     % cnt = cnt + 1;
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     %% Do Optimal Seq
%     cnt = 1;
%     for jj = 1:N
%         
%         child_seq = opt_seq(1:jj-1);
%         
%         PF_node = encode_sequence(child_seq,N);
%         
%         idx = length(child_seq)+1;
%         label = zeros(1,N);
%         label(opt_seq(idx)) = 1;
%         
%         
%         X(:,:,cnt) = [PF; PF_node];
%         Y(cnt,:) = label;
%         
%         cnt = cnt + 1;
%         
%     end
%     optimal_action = opt_seq(ii);
%     
%     parent_seq = [parent_seq; optimal_action];
%     
%     
% end

