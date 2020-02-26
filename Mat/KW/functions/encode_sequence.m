function PF_node = encode_sequence(seq,N)

% seq: Input Sequence
% N: number of tasks

PF_node = zeros(N,N);

if ~isempty(seq)
    
    col_idx(:,1) = seq;
    row_idx(:,1) = 1:length(seq);
    
    idx = sub2ind(size(PF_node),row_idx,col_idx);
    PF_node(idx) = 1;
%     PF_node(N+1,1:length(seq)) = seq; % Encode the past decisions as well
end