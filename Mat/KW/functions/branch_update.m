function B_new = branch_update(B,n,l_task,s_task,d_task,w,t_drop,l_drop,B_new,curTime)
%%% Generate New Branch

seq_prev = B.seq;
t_ex_prev = B.t_ex;
l_inc_prev = B.l_inc;


seq = [seq_prev; n];

t_ex = t_ex_prev;
if isempty(seq_prev)
    t_ex(n) = max([s_task(n); curTime]);
else
    t_ex(n) = max([s_task(n) ; t_ex_prev(seq_prev(end)) + d_task(seq_prev(end))]);
end


l_inc = l_inc_prev + l_task{n}(t_ex(n));


LB = l_inc;
UB = l_inc;

N = numel(l_task);
T_c_prev = B.Tc;
T_c = T_c_prev;
T_c(  T_c_prev == seq(end)) = [];
% T_c = setdiff((1:N)',seq);

t_end = t_ex(n) + d_task(n);
t_max = sum(d_task(T_c)) + max([s_task(T_c) ; t_end]);

if ~isempty(T_c)
    Tmax_vec = max( s_task(T_c') , t_end );
    cnt = 1;
    for n = T_c'        
        
        LB = LB + cost_linDrop(max([s_task(n) ; t_end]),w(n),s_task(n),t_drop(n),l_drop(n));        
%         LB = LB + l_task{n}(max([s_task(n) ; t_end]));
%         LB = LB + l_task{n}( Tmax_vec(cnt)  );  cnt = cnt + 1;
%         UB = UB + l_task{n}(t_max);
        UB = UB + cost_linDrop(t_max,w(n),s_task(n),t_drop(n),l_drop(n));
        
    end
end

B_new.seq = seq;
B_new.t_ex = t_ex;
B_new.l_inc = l_inc;
B_new.LB = LB;
B_new.UB = UB;
B_new.Tc = T_c;

% B_new = struct('seq',seq,'t_ex',t_ex,'l_inc',l_inc,'LB',LB,'UB',UB,'Tc',T_c);
% B_new = struct('seq',seq,'t_ex',t_ex,'l_inc',l_inc,'LB',LB,'UB',UB,'Tc',T_c);

