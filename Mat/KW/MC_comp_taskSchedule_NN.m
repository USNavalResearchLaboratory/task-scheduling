%%%%%%%%%% Task Scheduling Comparison

profile clear
profile on -history

addpath(genpath('.\functions'))

clear;
rng(108);


%%% Inputs
N_mc = 1000;

% Algorithms
% fcn_search = {@(s_task,d_task,l_task) fcn_BB(s_task,d_task,l_task,'LIFO');
%     @(s_task,d_task,l_task) fcn_MCTS_fast(s_task,d_task,l_task,1000)};
% fcn_search = {@(s_task,d_task,l_task) fcn_BB_NN(s_task,d_task,l_task,'LIFO');
%     @(s_task,d_task,l_task) fcn_MCTS(s_task,d_task,l_task,1000)};
fcn_search = {@(s_task,d_task,l_task) fcn_BB_NN(s_task,d_task,l_task,'LIFO');};

% Tasks
N = 3;                      % number of tasks
SAMPLE_THRESHOLD = 500000;

s_task = 100*rand(N,1);            % task start times
d_task = 2 + 9*rand(N,1);          % task durations
w = 1 + 4*rand(N,1);

t_drop = s_task + d_task.*(3+2*rand(N,1));
l_drop = (2+rand(N,1)).*w.*(t_drop-s_task);

l_task = cell(N,1);
for n = 1:N
    l_task{n} = @(t) cost_linDrop(t,w(n),s_task(n),t_drop(n),l_drop(n));
end





%%% Monte Carlo Simulation
N_alg = numel(fcn_search);

loss_mc = zeros(N_mc,N_alg);
t_run_mc = zeros(N_mc,N_alg);
X = [];
Y = [];
tstart = tic;
for i_mc = 1:N_mc
    
    if mod(i_mc,100) == 0
        fprintf('Task Set %i/%i , Samples = %i \n',i_mc,N_mc,size(X,3));
    end
    
    %     % Tasks
    %     N = 8;                      % number of tasks
    %
    s_task = 100*rand(N,1);            % task start times
    d_task = 2 + 9*rand(N,1);          % task durations
    w = 1 + 4*rand(N,1);
    
    
    t_drop = s_task + d_task.*(3+2*rand(N,1));
    l_drop = (2+rand(N,1)).*w.*(t_drop-s_task);
    
    l_task = cell(N,1);
    for n = 1:N
        l_task{n} = @(t) cost_linDrop(t,w(n),s_task(n),t_drop(n),l_drop(n));
    end
    
    
    % SearchX(
    for i_a = 1:N_alg
        [t_ex,loss,t_run,Xnow,Ynow] = fcn_search{i_a}(s_task,d_task,l_task);
        
        loss_mc(i_mc,i_a) = loss;
        t_run_mc(i_mc,i_a) = t_run;
        
        
        X = cat(3,X,Xnow);
        Y = [Y; Ynow];
        
    end
    
    if size(X,3) > SAMPLE_THRESHOLD
        break
    end
    
    if toc(tstart) > 60
        num_samps = size(X,3);
        samps_per_mc = num_samps/i_mc;
        fprintf('TIME = %f, Tasks Per MC = %f, Num Samples = %f \n',toc(tstart), samps_per_mc, num_samps)
        tstart = tic;
    end
    
end




%%% Results

% Plots

figure(1); clf; hold all;
for jj = 1:size(loss_mc,2)
    plot(t_run_mc(:,1),loss_mc(:,1),'.');
end
grid on; xlabel('Run Time'); ylabel('Loss'); legend('BB','MCTS');
title('Scheduler Performace for Random Task Sets');

profile viewer

%% NN implementations

try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end

Xinput = reshape(X,[size(X,1), size(X,2), 1, size(X,3)]);

Nsamps = size(Xinput,4);

clear Yclass
for jj = 1:Nsamps
    Yclass(jj,1) = find(Y(jj,:));
end
Yclass = categorical(Yclass);

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
    fullyConnectedLayer(N,'Name','fc_Out')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','class')];

% lgraph = layerGraph;
lgraph = layerGraph(layers);
figure(2); clf;
plot(lgraph)

% analyzeNetwork(layers)

MiniBatchSize = round(Ntrain*0.01);
NUM_BATCH_PER_EPOCH = round(Ntrain / MiniBatchSize);
VALID_PER_EPOCH = 4;
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
    'MiniBatchSize',MiniBatchSize, ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','auto');


net = trainNetwork(Xtrain,Ytrain,layers,options);


[YPred,scores] = classify(net,Xtest);

figure(3); clf;
cm = confusionchart(Ytest,YPred);

Acc = sum(diag( cm.NormalizedValues)) / sum( cm.NormalizedValues(:));
title(sprintf('Accuracy = %0.2f',Acc))


fname = ['NN_REPO/net_task_' num2str(N)  '_' datestr(now,30)];
save(fname,'net')



%% Neural Net Implemenation of Policy
profile clear
profile on -history

rng(182);
N_mc = 10;

% Tasks
% N = 20;                      % number of tasks

s_task = 30*rand(N,1);            % task start times
d_task = 1 + 2*rand(N,1);          % task durations


w = 0.8 + 10*rand(N,1);
t_drop = s_task + d_task.*(3+2*rand(N,1));
l_drop = (2+rand(N,1)).*w.*(t_drop-s_task);

l_task = cell(N,1);
for n = 1:N
    l_task{n} = @(t) cost_linDrop(t,w(n),s_task(n),t_drop(n),l_drop(n));
end





%%% Monte Carlo Simulation
fcn_search = {@(s_task,d_task,l_task) fcn_BB_NN(s_task,d_task,l_task,'LIFO');
    @(s_task,d_task,l_task) fcn_ES(s_task,d_task,l_task,1000)};
% fcn_search = {@(s_task,d_task,l_task) fcn_ES(s_task,d_task,l_task,1000)};



N_alg = numel(fcn_search);

loss_mc = zeros(N_mc,N_alg);
t_run_mc = zeros(N_mc,N_alg);
% X = [];
% Y = [];
for i_mc = 1:N_mc
    
    if mod(i_mc,100) == 0
        fprintf('Task Set %i/%i \n',i_mc,N_mc);
    end
    
    %     % Tasks
    %     N = 8;                      % number of tasks
    %
    
    s_task = 100*rand(N,1);            % task start times
    d_task = 2 + 9*rand(N,1);          % task durations
    w = 1 + 4*rand(N,1);
    
    
    t_drop = s_task + d_task.*(3+2*rand(N,1));
    l_drop = (2+rand(N,1)).*w.*(t_drop-s_task);
    
    l_task = cell(N,1);
    for n = 1:N
        l_task{n} = @(t) cost_linDrop(t,w(n),s_task(n),t_drop(n),l_drop(n));
        [~,~,~,~,c_drop(n)] = l_task{n}(0);
    end
    
    
    tic;
    [loss_mc(i_mc,N_alg + 1)] =  fcn_Inference_BB_NN(s_task,d_task,w,t_drop,c_drop,l_task,N,net); 
    t_run_mc(i_mc,N_alg + 1) = toc;
    
    
    MC = 50;
%     tic;    
    [loss_mc(i_mc,N_alg + 2)] =  fcn_Inference_MCTS_NN(MC,s_task,d_task,w,t_drop,c_drop,l_task,N,net); 
    t_run_mc(i_mc,N_alg + 2) = toc;
    
    
    
    % Search
    for i_a = 1:N_alg
        if i_a == 1
            [t_ex,loss,t_run,Xnow,Ynow] = fcn_search{i_a}(s_task,d_task,l_task);
        else
            tic;
            [t_ex,loss] = fcn_search{i_a}(s_task,d_task,l_task);
            t_run = toc;
        end
        
        loss_mc(i_mc,i_a) = loss;
        t_run_mc(i_mc,i_a) = t_run;
        %         X = cat(3,X,Xnow);
        %         Y = [Y; Ynow];
        
    end
    
end


%
%%% Results

% Plots
colors = ['b' 'r' 'g' 'k'];

figure(4); clf; hold all;
for jj = 1:size(loss_mc,2)
    plot(t_run_mc(:,jj),loss_mc(:,jj),['.' colors(jj)]);
end
grid on; xlabel('Run Time (MATLAB TIME)'); ylabel('Loss'); legend('BB','ES','NN','MCTS-NN');
title('Scheduler Performace for Random Task Sets');

for jj = 1:size(loss_mc,2)
    plot( mean(t_run_mc(:,jj)) , mean(loss_mc(:,jj)) ,['d' colors(jj)] ,'MarkerSize',10 ,'LineWidth',4 )
end


profile viewer

