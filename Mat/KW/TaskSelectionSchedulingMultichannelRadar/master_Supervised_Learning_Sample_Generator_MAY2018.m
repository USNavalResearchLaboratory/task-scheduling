
%% Algorithms Implemented here are based on "Task Selection and Scheduling in Multifunction Multichannel Radars" by M. Shaghaghi and R. Adve, 978-1-4673-8823-8/17/$31.00 ©2017 IEEE
clearvars


FLAG.profile = 1;

if FLAG.profile
    profile clear
    profile on -history
end

seed = 111;
rng(seed)

MONTE = 500000;
NumSamps = 100*1e3; 

FLAG.constrained = 1; % Confines input paramters to be much simplier start times all 0, task lengths all 5


% N = 8; % Number of Tasks
K = 1; % Number of identical channels or machines
% K = 1; % Number of identical channels or machines
% K = 4; % Number of identical channels or machines
T = 100; % Time window of all tasks


% N_vec = 20:5:30;
N_vec = 4;
% N_vec = 16;
% N_vec = 12;

NumN = length(N_vec);

% RunTime.EST = zeros(MONTE,NumN);
% RunTime.EstSwap = zeros(MONTE,NumN);
% RunTime.ED = zeros(MONTE,NumN);
% RunTime.EdSwap = zeros(MONTE,NumN);
% RunTime.BB = zeros(MONTE,NumN);
% 
% Cost.EST = zeros(MONTE,NumN);
% Cost.EstSwap = zeros(MONTE,NumN);
% Cost.ED = zeros(MONTE,NumN);
% Cost.EdSwap = zeros(MONTE,NumN);
% Cost.BB = zeros(MONTE,NumN);
% 
% DropPercent.EST = zeros(MONTE,NumN);
% DropPercent.EstSwap = zeros(MONTE,NumN);
% DropPercent.ED = zeros(MONTE,NumN);
% DropPercent.EdSwap = zeros(MONTE,NumN);
% DropPercent.BB = zeros(MONTE,NumN);


DATA = [];
AuxData = [];
X = [];
Y = [];

tstart = tic;
cnt.N = 1;
for N = N_vec
    for monte = 1:MONTE
        
        if mod(monte,10) == 0
            fprintf('Monte %i of %i, Samps Collected %i of %i, TIME ELAPSED = %f \n\n',monte,MONTE,size(X,3),NumSamps,toc(tstart))
        end
        
        if FLAG.constrained
            s_task = rand(N,1)*4; % Starting time of tasks
%             s_task = rand(N,1)*0.001;
            viable_task = 6*rand(N,1); % Difference between deadline d_n and starting time s_n
            deadline_task = viable_task + s_task; % Task deadline equal to the starting time + viablity window of each task
%             length_task = 5*ones(N,1)*1e-3;%rand(N,1)*9 + 2;  % Task processing length
%             drop_task = 0*( rand(N,1)*400 + 100 ); % Dropping cost of each task
            length_task = 36*ones(N,1)*1e-3;%rand(N,1)*9 + 2;  % Task processing length
            drop_task = 100*ones(N,1);%0*( rand(N,1)*400 + 100 ); % Dropping cost of each task
            
            %         w_task = rand(N,1)*4 + 1;   % Tardiness penalty of each task
            w_task = randi(10,[N,1]);
        else
            
            s_task = rand(N,1)*100; % Starting time of tasks
            viable_task = 10*rand(N,1) + 2; % Difference between deadline d_n and starting time s_n
            deadline_task = viable_task + s_task; % Task deadline equal to the starting time + viablity window of each task
            length_task = rand(N,1)*9 + 2;  % Task processing length
            drop_task = rand(N,1)*400 + 100; % Dropping cost of each task
            w_task = rand(N,1)*4 + 1;   % Tardiness penalty of each task
            
%             s_task = 100*rand(N,1);            % task start times
%             length_task = 2 + 9*rand(N,1);          % task durations
%             w_task = 1 + 4*rand(N,1);
%             deadline_task = s_task + length_task.*(3+2*rand(N,1));
%             drop_task = (2+rand(N,1)).*w_task.*(deadline_task-s_task);
        end
        
        
        % EST
        tic
        [~,T] = sort(s_task); % Sort jobs based on starting times
        [Cost.EST(monte,cnt.N),t_ex,NumDropTask] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
        %         [T,Cost.EST(monte,cnt.N),t_ex,NumDropTask] = EST_MultiChannel(N,K,s_task,w_task,deadline_task,length_task,drop_task);
        DropPercent.EST(monte,cnt.N) = NumDropTask/N;
        RunTime.EST(monte,cnt.N) = toc;
        
        % Perform Task Swapping for EST
        if NumDropTask > 0
            for jj = 1:N-1
                Tswap = T;
                T1 = T(jj);
                T2 = T(jj+1);
                Tswap(jj) = T2;
                Tswap(jj+1) = T1;
                [~,t_ex] = MultiChannelSequenceScheduler(Tswap,N,K,s_task,w_task,deadline_task,length_task,drop_task);
                if sum( t_ex < deadline_task ) == N
                    T = Tswap;
                    break
                end
            end
            [Cost.EstSwap(monte,cnt.N),t_ex,NumDropTask] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
            DropPercent.EstSwap(monte,cnt.N) = NumDropTask/N;
            RunTime.EstSwap(monte,cnt.N) = toc;
        else
            Cost.EstSwap(monte,cnt.N) = Cost.EST(monte,cnt.N);
            DropPercent.EstSwap(monte,cnt.N) = DropPercent.EST(monte,cnt.N);
            RunTime.EstSwap(monte,cnt.N) = RunTime.EST(monte,cnt.N);
            
        end

        
        % ED
        tic
        [~,T] = sort(deadline_task); % Sort jobs based on starting times
        [Cost.ED(monte,cnt.N),t_ex,NumDropTask] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
        DropPercent.ED(monte,cnt.N) = NumDropTask/N;
        RunTime.ED(monte,cnt.N) = toc;

        
        % Perform Task Swapping for ED
        if NumDropTask > 0
            for jj = 1:N-1
                Tswap = T;
                T1 = T(jj);
                T2 = T(jj+1);
                Tswap(jj) = T2;
                Tswap(jj+1) = T1;
                [~,t_ex] = MultiChannelSequenceScheduler(Tswap,N,K,s_task,w_task,deadline_task,length_task,drop_task);
                if sum( t_ex < deadline_task ) == N
                    T = Tswap;
                    break
                end
            end
            [Cost.EdSwap(monte,cnt.N),t_ex,NumDropTask] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
            DropPercent.EdSwap(monte,cnt.N) = NumDropTask/N;
            RunTime.EdSwap(monte,cnt.N) = toc;

        else
            Cost.EdSwap(monte,cnt.N) = Cost.ED(monte,cnt.N);
            DropPercent.EdSwap(monte,cnt.N) = DropPercent.ED(monte,cnt.N);
            RunTime.EdSwap(monte,cnt.N) = RunTime.ED(monte,cnt.N);
        end

        
        % B&B Scheduler 
%         rng(10)
        tic
        % Example 
%         N = 7; K = 3;
%         s_task = [0 0 0 3 0 0 0]';
%         length_task = [3 2 4 2 3 3 3]';
%         deadline_task = 100*ones(7,1); drop_task = 100*ones(7,1);
%         w_task = ones(7,1);        
%         [T,Tscheduled,Tdrop] = BBscheduler(K,s_task,deadline_task,length_task,drop_task,w_task);
        [T,Tscheduled,Tdrop,NodeStats] = BBschedulerWithStats(K,s_task,deadline_task,length_task,drop_task,w_task);    
%         [T2,~,~] = BBschedulerStack(K,s_task,deadline_task,length_task,drop_task,w_task);

        for kk = 1:length(NodeStats)
            NodeParams(kk).s_task = s_task;
            NodeParams(kk).deadline_task = deadline_task;
            NodeParams(kk).length_task = length_task;
            NodeParams(kk).drop_task = drop_task;
            NodeParams(kk).w_task = w_task;
        end
        

        CutOff = round(1000/N); % Number of branches to generate data for (smaller yeilds more monte carlos)

        [Xnow,Ynow] = SupervisedLearningDataGeneration(NodeStats,NodeParams(1),K,CutOff);
        X = cat(3,X,Xnow);
%         X = [X; Xnow];
        Y = [Y; Ynow];
        if ~all(Ynow > 0)
            keyboard
        end
        
%         DATA = [DATA NodeStats];
%         AuxData = [AuxData NodeParams];
        NodeParams = [];
%         if length(DATA) > NumSamps
%             break
%         end
               
        [Cost.BB(monte,cnt.N),t_ex,NumDropTask] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
        DropPercent.BB(monte,cnt.N) = NumDropTask/N;
        RunTime.BB(monte,cnt.N) = toc;

        if size(NodeStats,2) > 3
%             keyboard
        end
            
        
        if length(T) < N
            keyboard
        end
        
        if Cost.BB(monte,cnt.N) < 0
            keyboard
        end
           
        if any( Cost.BB(monte,cnt.N) > [Cost.EST(monte,cnt.N)  Cost.EstSwap(monte,cnt.N) Cost.ED(monte,cnt.N) Cost.EdSwap(monte,cnt.N)] )
            keyboard
        end
        
        if size(X,3) > NumSamps
            break
        end
        
        
    end
    cnt.N = cnt.N + 1;
end



figure(81); clf;
AAA = X(end-N+1:end,:,:);
hist(  squeeze(  sum(sum(AAA,1),2) )  )
xlabel('Number of Actions Taken in Training Sample')
ylabel('Distribution')




%% Generate Supervised Learning data and labels



% for jj = 1:length(DATA)
% 
%     curNode = DATA(jj);    
%     BestSeq = curNode.BestSeq;
%     
%     s_task = AuxData(jj).s_task;
%     deadline_task = AuxData(jj).deadline_task;
%     length_task = AuxData(jj).length_task;
%     drop_task = AuxData(jj).drop_task;
%     w_task = AuxData(jj).w_task;
%     
% %     [t_ex,x,ChannelAvailableTime,TaskChannel] = BBMultiChannelSequenceScheduler(T,s_task,deadline_task,length_task,ChannelAvailableTime);
% 
%     
%     for kk = 1:N
%         
%         node = BestSeq(1:kk-1);
%         optimal_action = BestSeq(kk);
%         
%         ChannelAvailableTime = zeros(K,1);
%         [t_ex,x,ChannelAvailableTime,TaskChannel] = BBMultiChannelSequenceScheduler(node,s_task,deadline_task,length_task,ChannelAvailableTime);
%         
%     end
% end



%% Train NN 

% % Normalize X by max value of 500
% feature_bound = size(X,1)-2*N;
% X(1:feature_bound,:,:) = X(1:feature_bound,:,:)/500;

try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end

Xinput = reshape(X,[size(X,1), size(X,2), 1, size(X,3)]);

Nsamps = size(Xinput,4);

clear Yclass
% for jj = 1:Nsamps
%     Yclass(jj,1) = find(Y(jj,:));
% end
% Yclass = categorical(Yclass);
Yclass = categorical(Y);
Nclass = length(unique(Yclass));

Ntrain = round(0.7*Nsamps);
Nvalid = round(0.15*Nsamps);
Ntest  = round(0.15*Nsamps);

idx = randperm(size(Xinput,4));
idx_train = idx(1:Ntrain);
idx_valid = idx(Ntrain+1:Ntrain+1+Nvalid);
idx_test =  idx(Ntrain+2+Nvalid:end);

Xtrain = Xinput(:,:,:,idx_train);
Ytrain = Yclass(idx_train);

Xvalid = Xinput(:,:,:,idx_valid);
Yvalid = Yclass(idx_valid);

Xtest = Xinput(:,:,:,idx_test);
Ytest = Yclass(idx_test);


% CNN_filters = 16;
% PadSize = 1;
% CnnSize = [ 2 2];

%%

CNN_filters = 16;

layers = [
    imageInputLayer([size(X,1) size(X,2) 1],'Name','Input','Normalization','zerocenter')
    
    convolution2dLayer([1 4],CNN_filters,'Name','conv1','Padding','Same')
    batchNormalizationLayer('Name','BN1')
    reluLayer('Name','relu')
    %     maxPooling2dLayer(2,'Stride',2,'Name','mp1')
    
    convolution2dLayer([1 4],CNN_filters,'Name','conv2','Padding','Same')
    batchNormalizationLayer('Name','BN2')
    reluLayer('Name','relu2')
    %     maxPooling2dLayer(2,'Stride',2,'Name','mp2')
    
    convolution2dLayer([1 4],CNN_filters,'Name','conv3','Padding','Same')
    batchNormalizationLayer('Name','BN3')
    reluLayer('Name','relu3')
    
    convolution2dLayer([1 4],CNN_filters,'Name','conv4','Padding','Same')
    batchNormalizationLayer('Name','BN4')
    reluLayer('Name','relu4')
    dropoutLayer('Name','Drop4')

    fullyConnectedLayer(1024,'Name','fc1')
    fullyConnectedLayer(128,'Name','fc2')
    fullyConnectedLayer(Nclass,'Name','fc_Out')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','class')];

% lgraph = layerGraph;
lgraph = layerGraph(layers);
figure(2); clf;
plot(lgraph)

% analyzeNetwork(layers)

MiniBatchSize = round(Ntrain*0.01);
NUM_BATCH_PER_EPOCH = round(Ntrain / MiniBatchSize);
VALID_PER_EPOCH = 2;
ValidationFrequency = round(NUM_BATCH_PER_EPOCH/VALID_PER_EPOCH);
% ValidationPatience =  20*VALID_PER_EPOCH;
ValidationPatience =  Inf;


options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',1000, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{Xvalid, Yvalid}, ...
    'ValidationFrequency',ValidationFrequency, ...
    'ValidationPatience',ValidationPatience, ...
    'VerboseFrequency', ValidationFrequency , ...
    'MiniBatchSize',MiniBatchSize, ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','auto');


net = trainNetwork(Xtrain,Ytrain,layers,options);


[YPred,scores] = classify(net,Xtest);

figure(3); clf;
cm = confusionchart(Ytest,YPred);

Acc = sum(diag( cm.NormalizedValues)) / sum( cm.NormalizedValues(:));
title(sprintf('Accuracy = %0.2f',Acc))


fname = ['NN_REPO/net_task_' num2str(N) '_K_' num2str(K) '_Filter' num2str(CNN_filters) '_' datestr(now,30)];
save(fname,'net')



%% Graphics
leg_str{1} = 'EST';
leg_str{2} = 'EST Swap';
leg_str{3} = 'ED';
leg_str{4} = 'ED Swap';
leg_str{5} = 'BB';
color_shape{1}= 'bv';
color_shape{2}= 'rs';
color_shape{3}= 'gd';
color_shape{4}= 'm*';
color_shape{5}= 'co';




figure(1); clf; hold all; grid on;
plot(mean(RunTime.EST,1),mean(Cost.EST,1),color_shape{1},'MarkerSize',12,'LineWidth',3)
plot(mean(RunTime.EstSwap,1),mean(Cost.EstSwap,1),color_shape{2},'MarkerSize',12,'LineWidth',3)
plot(mean(RunTime.ED,1),mean(Cost.ED,1),color_shape{3},'MarkerSize',12,'LineWidth',3)
plot(mean(RunTime.EdSwap,1),mean(Cost.EdSwap,1),color_shape{4},'MarkerSize',12,'LineWidth',3)
plot(mean(RunTime.BB,1),mean(Cost.BB,1),color_shape{5},'MarkerSize',12,'LineWidth',3)
legend(leg_str)
xlabel(sprintf('RunTime (seconds) (K = %i Channels)', K )) 
ylabel(sprintf('Average Cost (K = %i Channels)' , K ))

% plot((RunTime.EST),(Cost.EST),color_shape{1})
% plot((RunTime.EstSwap),(Cost.EstSwap),color_shape{2})
% plot((RunTime.ED),(Cost.ED),color_shape{3})
% plot((RunTime.EdSwap),(Cost.EdSwap),color_shape{4})
% plot((RunTime.BB),(Cost.BB),color_shape{5})




% figure(1); 
% subplot(2,2,1)
% cla; hold all; grid on;
% plot(N_vec,100*(1-mean(DropPercent.EST,1)),'-v')
% plot(N_vec,100*(1-mean(DropPercent.EstSwap,1)),'-s')
% plot(N_vec,100*(1-mean(DropPercent.ED,1)),'-d')
% plot(N_vec,100*(1-mean(DropPercent.EdSwap,1)),'-o')
% plot(N_vec,100*(1-mean(DropPercent.BB,1)),'-o')
% 
% 
% xlabel('Number of Tasks')
% ylabel('Percent of tasks scheduled')
% title(sprintf('Monte = %i',MONTE))
% legend(leg_str)
% 
% 
% % figure(2); 
% subplot(2,2,2)
% cla; hold all; grid on
% plot(N_vec,mean(Cost.EST,1),'-v')
% plot(N_vec,mean(Cost.EstSwap,1),'-s')
% plot(N_vec,mean(Cost.ED,1),'-d')
% plot(N_vec,mean(Cost.EdSwap,1),'-o')
% plot(N_vec,mean(Cost.BB,1),'-o')
% 
% 
% xlabel('Number of Tasks')
% ylabel('Average Cost (K = 4 Channels)')
% legend(leg_str)
% 
% 
% % figure(3); 
% subplot(2,2,[3:4])
% cla; grid on; hold all;
% plot(N_vec,mean(RunTime.EST,1),'-v')
% plot(N_vec,mean(RunTime.EstSwap,1),'-s')
% plot(N_vec,mean(RunTime.ED,1),'-d')
% plot(N_vec,mean(RunTime.EdSwap,1),'-o')
% plot(N_vec,mean(RunTime.BB,1),'-o')
% xlabel('Number of Tasks')
% ylabel(sprintf('RunTime (seconds) (K = %i Channels)', K )) 
% legend(leg_str)

%% Implementation and Comparison


MONTE = 10;
% NumSamps = 1*1e3; 
clear Cost DropPercent RunTime

tstart = tic;
cnt.N = 1;
for N = N_vec
    for monte = 1:MONTE
        
        if mod(monte,1) == 0
            fprintf('Monte %i of %i,  TIME ELAPSED = %f \n\n',monte,MONTE,toc(tstart))
        end
        
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
        
        
        % EST
        tic
        [~,T] = sort(s_task); % Sort jobs based on starting times
        [Cost.EST(monte,cnt.N),t_ex,NumDropTask] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
        %         [T,Cost.EST(monte,cnt.N),t_ex,NumDropTask] = EST_MultiChannel(N,K,s_task,w_task,deadline_task,length_task,drop_task);
        DropPercent.EST(monte,cnt.N) = NumDropTask/N;
        RunTime.EST(monte,cnt.N) = toc;
        
        % Perform Task Swapping for EST
        if NumDropTask > 0
            for jj = 1:N-1
                Tswap = T;
                T1 = T(jj);
                T2 = T(jj+1);
                Tswap(jj) = T2;
                Tswap(jj+1) = T1;
                [~,t_ex] = MultiChannelSequenceScheduler(Tswap,N,K,s_task,w_task,deadline_task,length_task,drop_task);
                if sum( t_ex < deadline_task ) == N
                    T = Tswap;
                    break
                end
            end
            [Cost.EstSwap(monte,cnt.N),t_ex,NumDropTask] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
            DropPercent.EstSwap(monte,cnt.N) = NumDropTask/N;
            RunTime.EstSwap(monte,cnt.N) = toc;
        else
            Cost.EstSwap(monte,cnt.N) = Cost.EST(monte,cnt.N);
            DropPercent.EstSwap(monte,cnt.N) = DropPercent.EST(monte,cnt.N);
            RunTime.EstSwap(monte,cnt.N) = RunTime.EST(monte,cnt.N);
            
        end

        
        % ED
        tic
        [~,T] = sort(deadline_task); % Sort jobs based on starting times
        [Cost.ED(monte,cnt.N),t_ex,NumDropTask] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
        DropPercent.ED(monte,cnt.N) = NumDropTask/N;
        RunTime.ED(monte,cnt.N) = toc;

        
        % Perform Task Swapping for ED
        if NumDropTask > 0
            for jj = 1:N-1
                Tswap = T;
                T1 = T(jj);
                T2 = T(jj+1);
                Tswap(jj) = T2;
                Tswap(jj+1) = T1;
                [~,t_ex] = MultiChannelSequenceScheduler(Tswap,N,K,s_task,w_task,deadline_task,length_task,drop_task);
                if sum( t_ex < deadline_task ) == N
                    T = Tswap;
                    break
                end
            end
            [Cost.EdSwap(monte,cnt.N),t_ex,NumDropTask] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
            DropPercent.EdSwap(monte,cnt.N) = NumDropTask/N;
            RunTime.EdSwap(monte,cnt.N) = toc;

        else
            Cost.EdSwap(monte,cnt.N) = Cost.ED(monte,cnt.N);
            DropPercent.EdSwap(monte,cnt.N) = DropPercent.ED(monte,cnt.N);
            RunTime.EdSwap(monte,cnt.N) = RunTime.ED(monte,cnt.N);
        end

        
        % B&B Scheduler 
%         rng(10)
        tic
        [T,Tscheduled,Tdrop,NodeStats] = BBschedulerWithStats(K,s_task,deadline_task,length_task,drop_task,w_task);
                       
               
        [Cost.BB(monte,cnt.N),t_ex,NumDropTask] = MultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task);
        DropPercent.BB(monte,cnt.N) = NumDropTask/N;
        RunTime.BB(monte,cnt.N) = toc;
                    
        if length(T) < N
            keyboard
        end
        
        if Cost.BB(monte,cnt.N) < 0
            keyboard
        end
           
        if any( Cost.BB(monte,cnt.N) > [Cost.EST(monte,cnt.N)  Cost.EstSwap(monte,cnt.N) Cost.ED(monte,cnt.N) Cost.EdSwap(monte,cnt.N)] )
            keyboard
        end
     
        % Policy Neural Net Implementation
        tic
        PF(1,:) = s_task;
        PF(2,:) = deadline_task;
        PF(3,:) = length_task;
        PF(4,:) = drop_task;
        PF(5,:) = w_task;
        
        PFtree = zeros(N,N);              
        PfStatus = zeros(3,N);
        PfStatus(1,:) = 1;
%         for nn = 1:length(node)
%             PfStatus(: , node(nn) ) = [0; 0; 1]; % Infeasible Already Assigned
%         end

        Xin = [PF; PFtree; PfStatus];
        node = zeros(N,1);
        for kk = 1:N
            [YPred,scores] = classify(net,Xin);
            scores(node(node ~= 0)) = 0;
            scores = scores/(sum(scores));
            [~,YPred] = max(scores);
            node(kk) = double(YPred);
            
            PFtree = zeros(N,N);
            IND = sub2ind([N N],[1:kk]',node(1:kk));
            PFtree(IND) = 1; 
            
            PfStatus = zeros(3,N);
            PfStatus(1,:) = 1;
            for nn = 1:kk
                PfStatus(: , node(nn) ) = [0; 0; 1]; % Infeasible Already Assigned 
            end
            Xin = [PF; PFtree; PfStatus];

        end
        
        [Cost.NN(monte,cnt.N),t_ex,NumDropTask] = MultiChannelSequenceScheduler(node,N,K,s_task,w_task,deadline_task,length_task,drop_task);
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
