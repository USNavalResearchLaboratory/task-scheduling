function node = generate_node_statistics_FAST(S_all,N,seq_opt,plot_en)




UB = cell2mat({S_all.UB});
LB = cell2mat({S_all.LB});

if plot_en
    figure(1); clf;
    hold all
    for jj = 1:length(UB)
        plot([jj jj],[LB(jj) UB(jj)],'-x')
    end
    grid on
end

for jj = 1:length(S_all)
    seq_length(jj) = length(S_all(jj).seq);
end


[~,sort_idx] = sort(seq_length);
S_sort = S_all(sort_idx);
seq_length = seq_length(sort_idx);
seq_mat = zeros(length(S_sort),N);
for jj = 1:length(S_sort)
    seq_mat(jj,:) = [S_sort(jj).seq' zeros(1,N-seq_length(jj))];
    UB(jj) = S_sort(jj).UB;
    LB(jj) = S_sort(jj).LB;
end


ALL_COMPLETE_SOLUTION_IDX = find(  seq_length == N );
BB = seq_mat( ALL_COMPLETE_SOLUTION_IDX ,:);


node_cnt = 1;
for jj = 1:length(S_sort)
    
    seq_cand = S_sort(jj).seq';
    seq_L = length(seq_cand);
    if isempty(seq_cand) % Root nodes: all nodes are descendents
        [~,min_idx] = min(UB);
        child_index = find(seq_length == 1);
        dom_index = child_index( LB(child_index) > min(UB(child_index)));
        dom_task = seq_mat(dom_index,1);
        
        node(node_cnt).seq = seq_cand;
        node(node_cnt).opt_a = S_sort(min_idx).seq(1);
        node(node_cnt).DOM = dom_task;
%         node(node_cnt).ND = 1; % Not Dominated Flag
        node_cnt = node_cnt + 1;
    else
        
        % Find descendents        
        offset = seq_cand - BB( : ,1:seq_L);        
        descend_idx = ALL_COMPLETE_SOLUTION_IDX( sum(offset == 0,2) == seq_L );  
        
        if ~isempty(descend_idx)
            child_index = descend_idx;
            dom_index = child_index( LB(child_index) > min(UB(child_index)));
            try
                dom_task  = [];

%                 if seq_L ~=N
%                     dom_task = unique(seq_mat(dom_index,seq_L+1));
%                 else
%                     dom_task  = [];
%                 end
            catch
                keyboard
            end
            
%             keyboard
            
            
            
            self_idx = find(descend_idx == jj);
            descend_idx(self_idx) = [];
            if ~isempty(descend_idx) % Make sure there are descendants
                try
                    [min_UB_all,min_idx] = min(UB(descend_idx));
                catch
                    keyboard
                end
                    
                    [min_UB,min_idx] = min(UB(descend_idx));
                    opt_a = S_sort(descend_idx(min_idx)).seq(seq_L+1);
                    A = [seq_mat(descend_idx,:), LB(descend_idx)', UB(descend_idx)']; % matrix with sequences (zero-padded and UB)];
                    
                    
                    if min_UB ~= min_UB_all
                        keyboard
                    end
                    
                    node(node_cnt).seq = seq_cand;
                    node(node_cnt).opt_a = opt_a;
                    node(node_cnt).DOM = dom_task;  
                    node_cnt = node_cnt + 1;              
                
            end
        
%         child_index = descend_idx(seq_length(descend_idx) == (seq_L+1));
%         dom_index = child_index( LB(child_index) > min(UB(child_index)));
%         try
%             if seq_L ~=N
%                 dom_task = seq_mat(dom_index,seq_L+1);
%             else
%                 dom_task  = [];
%             end
%         catch
%            keyboard 
%         end
%         
% %         L2 = zeros(size(offset,1),1);
% %         for kk = 1:length(offset)
% %             L2(kk) = sqrt(offset(kk,:)*offset(kk,:)');
% %         end
% %         descend_idx = find(L2 == 0);
%         self_idx = find(descend_idx == jj);
%         descend_idx(self_idx) = [];
%         if ~isempty(descend_idx) % Make sure there are descendants
%             try
%             [min_UB_all,min_idx] = min(UB(descend_idx));
%             catch
%                 keyboard
%             end
%             
%             complete_solution_flag = ( max(seq_length(descend_idx)) == N );
%             complete_solution_idx = find( (seq_length(descend_idx)) == N );   
%             descend_idx = descend_idx(complete_solution_idx);
%             
%             
%             
%             if complete_solution_flag
%                 
%                 
%                 [min_UB,min_idx] = min(UB(descend_idx));
%                 opt_a = S_sort(descend_idx(min_idx)).seq(seq_L+1);
%                 A = [seq_mat(descend_idx,:), LB(descend_idx)', UB(descend_idx)']; % matrix with sequences (zero-padded and UB)];
%                 
%                 
%                 if min_UB ~= min_UB_all
%                     keyboard
%                 end
% %                 min_idx ~=  
%                 
%                 node(node_cnt).seq = seq_cand;
%                 node(node_cnt).opt_a = opt_a; 
%                 node(node_cnt).DOM = dom_task;
%                 
% %                 ND_flag = (    sum( (seq_opt(1:seq_L) - seq_cand) == 0 ) == seq_L  );
% % %                 ND_flag = 0;
% %                 if ND_flag
% %                     node(node_cnt).ND = 1;
% %                 else
% %                     node(node_cnt).ND = 0;
% %                 end
%                 node_cnt = node_cnt + 1;
%             else % Descendents don't have a complete solution and can't evauluate optimal path
% %                 keyboard                
%             end

        end
            
%         descend_idx = find( norm( seq_cand - seq_mat(:,1:seq_L) ) == 0 )
        
    end
    
end



% term_idx = find(seq_length == N);
% 
% for jj = 1:length(term_idx)
%     
%     
%     
% end



% plot(1:length(LB),LB,'rx')
% plot(1:length(LB),UB,'bo')