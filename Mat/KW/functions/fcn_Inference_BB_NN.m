function [loss_mc] =  fcn_Inference_BB_NN(s_task,d_task,w,t_drop,c_drop,l_task,N,net)



tic;
PF = [];
PF(1,:) = s_task; % Start time of task
PF(2,:) = d_task; % Task Duration
PF(3,:) = w; % Task Slope
PF(4,:) = t_drop; % Task Drop Time
PF(5,:) = c_drop; % Task Drop Cost
%     PF_feasible = ones(1,N);
PF_status = zeros(3,N);
PF_status(1,:) = 1;

PF = [PF; PF_status; zeros(N,N)];

seq = [];
t_ex = zeros(N,1);
l_inc_prev = 0;
for n = 1:N
    [policy,scores] = classify(net,PF);
    
    one_hot = zeros(1,N);
    one_hot(policy) = 1;
    
    PF( end-N+n ,:,1) = one_hot;
    seq = [seq; double(policy)];
    PF(N+1,seq) = 0;
    PF(N+3,seq) = 1;
    
    
    if n == 1
        t_ex(n) = s_task(seq(n));
    else
        t_ex(n) = max([s_task(seq(n)) ; t_ex(n-1) + d_task(seq(n-1))]);
    end
    %         t_ex =
    l_inc = l_inc_prev + l_task{seq(n)}(t_ex(n));
    l_inc_prev = l_inc;
    
end
loss_mc = l_inc;
% t_run_mc(i_mc,N_alg + 1) = toc;