function B_new = branch_update_linear(B,n,w_task,s_task,d_task,B_new,curTime)
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


% l_inc = l_inc_prev + l_task{n}(t_ex(n));
l_inc = l_inc_prev + cost_linear(t_ex(n),w_task(n),s_task(n));


LB = l_inc;
UB = l_inc;

N = numel(s_task);
T_c_prev = B.Tc;
T_c = T_c_prev;
T_c(  T_c_prev == seq(end)) = [];
% T_c = setdiff((1:N)',seq);

t_end = t_ex(n) + d_task(n);
t_max = sum(d_task(T_c)) + max([s_task(T_c) ; t_end]);

if ~isempty(T_c)
    Tmax_vec = max( s_task(T_c') , t_end );
    cnt = 1;
    
    
    % Vectorized Operations for speed
    inputTime = max(s_task(T_c') , t_end);
    LB = LB + sum( cost_linear(inputTime,w_task(T_c),s_task(T_c)) );
    UB = UB + sum( cost_linear(t_max,w_task(T_c),s_task(T_c))  );
    
%     for n = T_c'        
%         LB = LB + cost_linear(max([s_task(n) ; t_end]),w_task(n),s_task(n));
%         UB = UB + cost_linear(t_max,w_task(n),s_task(n));        
%         
% %         LB = LB + cost_linDrop(max([s_task(n) ; t_end]),w(n),s_task(n),t_drop(n),l_drop(n));
% %         UB = UB + cost_linDrop(t_max,w(n),s_task(n),t_drop(n),l_drop(n));
%        
% %         LB = LB + l_task{n}(max([s_task(n) ; t_end]));
% %         LB = LB + l_task{n}( Tmax_vec(cnt)  );  cnt = cnt + 1;
% %         UB = UB + l_task{n}(t_max);
%         
%     end
end

B_new.seq = seq;
B_new.t_ex = t_ex;
B_new.l_inc = l_inc;
B_new.LB = LB;
B_new.UB = UB;
B_new.Tc = T_c;

% B_new = struct('seq',seq,'t_ex',t_ex,'l_inc',l_inc,'LB',LB,'UB',UB,'Tc',T_c);
% B_new = struct('seq',seq,'t_ex',t_ex,'l_inc',l_inc,'LB',LB,'UB',UB,'Tc',T_c);

