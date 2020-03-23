clearvars
seed = 111;
rng(seed)
FLAG.profile = 1;
FLAG.constrained = 1;
if FLAG.profile
    profile clear
    profile on -history
end

addpath(genpath('C:\Users\wagnerk\Desktop\WU_6B72_CRM\CRM_REPO\Mat\KW'))




%% Implementation and Comparison


MONTE = 10;
% NumSamps = 1*1e3; 
clear Cost DropPercent RunTime
tstart = tic;
cnt.N = 1;
N_vec = 6;
K = 1;




for N = N_vec
    % Load Required Neural Network
    clear net
%     NNstring = sprintf('net_task_%i_K_%i_FINAL.mat',N,K);
%     load(NNstring)
    
    NNstring = sprintf('net_task_%i_K_%i_FINAL.mat',4,K);
    load(NNstring);
    for monte = 1:MONTE
        
        if mod(monte,1) == 0
            fprintf('Monte %i of %i,  TIME ELAPSED = %f \n\n',monte,MONTE,toc(tstart))
        end
        
        % Generate Monte Carlo Data
        if FLAG.constrained
            s_task = rand(N,1)*0; % Starting time of tasks
            viable_task = 10*rand(N,1) + 2; % Difference between deadline d_n and starting time s_n
            deadline_task = viable_task + s_task; % Task deadline equal to the starting time + viablity window of each task
            length_task = 5*ones(N,1)*1e-3;%rand(N,1)*9 + 2;  % Task processing length
            drop_task = 0*( rand(N,1)*400 + 100 ); % Dropping cost of each task
            %         w_task = rand(N,1)*4 + 1;   % Tardiness penalty of each task
            w_task = randi(25,[N,1]);
        else
            s_task = 100*rand(N,1);            % task start times
            length_task = 2 + 9*rand(N,1);          % task durations
            w_task = 1 + 4*rand(N,1);
            deadline_task = s_task + length_task.*(3+2*rand(N,1));
            drop_task = (2+rand(N,1)).*w_task.*(deadline_task-s_task);
        end
        data.N = N;
        data.K = K;
        data.s_task = s_task; 
        data.w_task = w_task;
        data.deadline_task = deadline_task;
        data.length_task = length_task;
        data.drop_task = drop_task;
        
        
        % EST
        tic
        [Cost.EST(monte,cnt.N),t_ex,NumDropTask,T] = ESTalgorithm(data);        
        DropPercent.EST(monte,cnt.N) = NumDropTask/N;
        RunTime.EST(monte,cnt.N) = toc;
        
        % EST with Task Swapping
        tic
        [Cost.EstSwap(monte,cnt.N),t_ex,NumDropTask,T] = EstTaskSwapAlgorithm(data);
        DropPercent.EstSwap(monte,cnt.N) = NumDropTask/N;
        RunTime.EstSwap(monte,cnt.N) = toc;
             
        % ED
        tic
        [Cost.ED(monte,cnt.N),t_ex,NumDropTask,T] = EdAlgorithm(data);
        DropPercent.ED(monte,cnt.N) = NumDropTask/N;
        RunTime.ED(monte,cnt.N) = toc;

        % ED with Task Swapping
        tic
        [Cost.EdSwap(monte,cnt.N),t_ex,NumDropTask,T] = EdTaskSwapAlgorithm(data);
        DropPercent.EdSwap(monte,cnt.N) = NumDropTask/N;
        RunTime.EdSwap(monte,cnt.N) = toc;
              

        
        % B&B Scheduler 
%         rng(10)
        tic
        [Cost.BB(monte,cnt.N),t_ex,NumDropTask,T] = BranchBoundAlgorithm(data);
        DropPercent.BB(monte,cnt.N) = NumDropTask/N;
        RunTime.BB(monte,cnt.N) = toc;
                    
        if length(T) < N
            keyboard
        end
        
        if Cost.BB(monte,cnt.N) < 0
            keyboard
        end
           
        if any( Cost.BB(monte,cnt.N) > [Cost.EST(monte,cnt.N)  Cost.EstSwap(monte,cnt.N) Cost.ED(monte,cnt.N) Cost.EdSwap(monte,cnt.N)] )
            keyboard % Means BB is not optimal!!! What happened??
        end
     
       
     
        % MCTS with Policy NN
        tic
        M = 10; % Number of roll-outs
        [Cost.MCTS(monte,cnt.N),t_ex,NumDropTask,T] = MctsNeuralNetSchedulerAlgorithm(data,net,M);
        DropPercent.MCTS(monte,cnt.N) = NumDropTask/N;
        RunTime.MCTS(monte,cnt.N) = toc;
        
        
        % Policy Neural Net Implementation
        tic
        [Cost.NN(monte,cnt.N),t_ex,NumDropTask,T] = NeuralNetSchedulerAlgorithm(data,net);
        DropPercent.NN(monte,cnt.N) = NumDropTask/N;
        RunTime.NN(monte,cnt.N) = toc;
        
        
        
    end
    cnt.N = cnt.N + 1;
end



%% 
leg_str{1} = 'EST';
leg_str{2} = 'EST Swap';
leg_str{3} = 'ED';
leg_str{4} = 'ED Swap';
leg_str{5} = 'BB';
leg_str{6} = 'NN';
color_shape{1}= 'bv';
color_shape{2}= 'rs';
color_shape{3}= 'gd';
color_shape{4}= 'm*';
color_shape{5}= 'co';
color_shape{6} = 'k^';


figure(22); clf; hold all; grid on;
plot(mean(RunTime.EST,1),mean(Cost.EST,1),color_shape{1},'MarkerSize',12,'LineWidth',3)
plot(mean(RunTime.EstSwap,1),mean(Cost.EstSwap,1),color_shape{2},'MarkerSize',12,'LineWidth',3)
plot(mean(RunTime.ED,1),mean(Cost.ED,1),color_shape{3},'MarkerSize',12,'LineWidth',3)
plot(mean(RunTime.EdSwap,1),mean(Cost.EdSwap,1),color_shape{4},'MarkerSize',12,'LineWidth',3)
plot(mean(RunTime.BB,1),mean(Cost.BB,1),color_shape{5},'MarkerSize',12,'LineWidth',3)
plot(mean(RunTime.NN,1),mean(Cost.NN,1),color_shape{6},'MarkerSize',12,'LineWidth',3)

legend(leg_str)
xlabel(sprintf('RunTime (seconds) (K = %i Channels)', K )) 
ylabel(sprintf('Average Cost (K = %i Channels)',K))


%%

if FLAG.profile
    profile viewer
end
