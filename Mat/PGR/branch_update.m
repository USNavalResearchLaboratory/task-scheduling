function B_new = branch_update(B,n,l_task,s_task,d_task)
%%% Generate New Branch

seq_prev = B.seq;
t_ex_prev = B.t_ex;
l_inc_prev = B.l_inc;


seq = [seq_prev; n];

t_ex = t_ex_prev;
if isempty(seq_prev)
    t_ex(n) = s_task(n);
else
    t_ex(n) = max([s_task(n) ; t_ex_prev(seq_prev(end)) + d_task(seq_prev(end))]);
end


l_inc = l_inc_prev + l_task{n}(t_ex(n));


LB = l_inc;
UB = l_inc;

N = numel(l_task);
T_c = setdiff((1:N)',seq);

t_end = t_ex(n) + d_task(n);
t_max = sum(d_task(T_c)) + max([s_task(T_c) ; t_end]);

for n = T_c'
    LB = LB + l_task{n}(max([s_task(n) ; t_end]));
    UB = UB + l_task{n}(t_max);
end


B_new = struct('seq',seq,'t_ex',t_ex,'l_inc',l_inc,'LB',LB,'UB',UB);

