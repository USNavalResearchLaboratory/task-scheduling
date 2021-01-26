function [loss_mc] =  fcn_Inference_MCTS_NN_linear(MC,s_task,d_task,w_task,N,net)



tic;
PF = [];
PF(1,:) = s_task; % Start time of task
PF(2,:) = d_task; % Task Duration
PF(3,:) = w_task; % Task Slope
% PF(4,:) = t_drop; % Task Drop Time
% PF(5,:) = c_drop; % Task Drop Cost
%     PF_feasible = ones(1,N);
NumFeature = size(PF,1);
PF_status = zeros(3,N);
PF_status(1,:) = 1;
PF = [PF; PF_status; zeros(N,N)];



seq = [];
t_ex = zeros(N,1);
l_inc_prev = 0;

UB = Inf;
T = [];
ND = 1:N;
D = [];
Cstar = Inf;
Tstar = [];
comp_flag = false;
s = struct('T',T,'ND',ND,'D',D,'comp_flag',comp_flag,'Cstar',Cstar,'Tstar',Tstar,'PF',PF);
l_inc_prev = 0;


NetData = gpuArray(s.PF);

% [~,prior_policy] = classify(net,NetData,'Acceleration','auto'); 
[~,prior_policy] = classify(net,s.PF,'Acceleration','auto','ExecutionEnvironment','gpu'); 
% [aaa] = predict(net,s.PF,'Acceleration','mex'); 

seq_mat = zeros(N,1);

while ~isempty(s.ND)
    
    % Simulation Phase
    for mm = 1:MC
        seq = zeros(N,1);
        sprime = s;
        n = length(sprime.T) + 1;
        seq(1:n-1) = sprime.T;
        while ~isempty(sprime.ND) % Expand the rollout-node
            n = length(sprime.T) + 1;
            
            % This is very slow --> Store old probability calculations to
            % reduce calls
            
            
            [max_a,scores] = classify(net,sprime.PF,'Acceleration','auto','ExecutionEnvironment','gpu');

%             NetData = gpuArray( sprime.PF );
%             [max_a,scores] = classify(net,NetData,'Acceleration','auto','ExecutionEnvironment','gpu');  
%             [scores] = predict(net,sprime.PF,'Acceleration','auto','ExecutionEnvironment','gpu','ReturnCategorical',false);
%             [max_a] = predict(net,sprime.PF,'Acceleration','auto','ExecutionEnvironment','gpu','ReturnCategorical',true);
%             [~,max_a] = max(scores);
            
            
            scores(sprime.T) = []; % Remove scores corresponding to previously scheduled tasks
            scores = scores/sum(scores); % Renormalize
            cdf = cumsum(scores);
            cdf(end) = [];
            cdf = [0 cdf];
            rand_val = rand;
            task = sprime.ND( find( cdf > rand_val ,1,'first' )-1  );
            if isempty(task)
                task = sprime.ND(end);
            end
            try
                seq(n) = task;
            catch
                keyboard
            end
            if sum( sprime.ND == seq(n)) == 1   % Make sure task is in not-dominated set
                sprime.ND(sprime.ND == seq(n)) = [];
                sprime.T = [sprime.T seq(n)];
                n = length(sprime.T);
                
                
                one_hot = zeros(1,N);
                one_hot(seq(n)) = 1;  
                PF = sprime.PF;
                PF( end-N+n ,:,1) = one_hot;
                PF(NumFeature+1,seq(n)) = 0;
                PF(NumFeature+3,seq(n)) = 1;
                
                sprime.PF = PF;
                
            else
                keyboard  
            end
        end
        
        l_inc = 0;
        l_inc_prev = 0;
        t_ex = zeros(N,1);
        for n = 1:N
            if n == 1
                t_ex(n) = s_task(seq(n));
            else
                t_ex(n) = max([s_task(seq(n)) ; t_ex(n-1) + d_task(seq(n-1))]);
            end
            l_inc = l_inc_prev + cost_linear(t_ex(n),w_task(seq(n)),s_task(seq(n)));
%             l_inc = l_inc_prev + l_task{seq(n)}(t_ex(n));
            l_inc_prev = l_inc;
        end
        
        if l_inc < s.Cstar
            s.Cstar = l_inc;
            s.Tstar = seq;
        end  
    end
    
    % Decision Phase (Given results of MCTS perform selection)
    n = length(s.T) + 1;
    task = s.Tstar(n);
    s.T = [s.T task];
    s.ND( s.T(end) == s.ND ) = [];
    
    seq = s.T;
    one_hot = zeros(1,N);
    one_hot(seq(n)) = 1;
    PF = s.PF;
    PF( end-N+n ,:,1) = one_hot;
    PF(NumFeature+1,seq(n)) = 0;
    PF(NumFeature+3,seq(n)) = 1;
    
    s.PF = PF;
    
    
    
%     s     
%     aa = 1;    
%     seq = s.Tstar;
%     for n = 1:N
%         if n == 1
%             t_ex(n) = s_task(seq(n));
%         else
%             t_ex(n) = max([s_task(seq(n)) ; t_ex(n-1) + d_task(seq(n-1))]);
%         end        
%         C(n) = l_task{seq(n)}(t_ex(n));
%     end
    
end


seq = s.Tstar;
l_inc_prev = 0;
l_inc = 0;
t_ex = zeros(N,1);
for n = 1:N    
    if n == 1
        t_ex(n) = s_task(seq(n));
    else
        t_ex(n) = max([s_task(seq(n)) ; t_ex(n-1) + d_task(seq(n-1))]);
    end
    %         t_ex =
    l_inc = l_inc_prev + cost_linear(t_ex(n),w_task(seq(n)),s_task(seq(n)));
%     l_inc = l_inc_prev + l_task{seq(n)}(t_ex(n));
    l_inc_prev = l_inc;
    
end
loss_mc = l_inc;
% t_run_mc(i_mc,N_alg + 1) = toc;