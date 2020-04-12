%% Description
% This code sequentially assigns jobs as resource periods (RPs) become
% available. For example: at time t = 0 assigns and execute N tasks, while
% t < N*RP re-evaluate assignments for subsequent jobs 

clearvars
close all

FLAG.profile = 1;
FLAG.save = 0;
FLAG.check = 0;
FLAG.FixedPriority = 0; % Used to keep same inputs to sequence-schedulers for all algorithms
                        % Something is broken here. EST is doing better
                        % than BB, but checked for same inputs get same
                        % answer.
if FLAG.profile
    profile clear
    profile on -history
end

seed = 12307;
rng(seed)

addpath('./functions/')
addpath('./TaskSelectionSchedulingMultichannelRadar/')


cnt_apr = 1;
approach_string{cnt_apr} = 'EST'; cnt_apr = cnt_apr + 1;
approach_string{cnt_apr} = 'BB'; cnt_apr = cnt_apr + 1;
% approach_string{cnt_apr} = 'ED'; cnt_apr = cnt_apr + 1;
% approach_string{cnt_apr} = 'NN'; cnt_apr = cnt_apr + 1;


% approach_string{2} = 'NN_Single';
% approach_string{2} = 'MCTS';
% approach_string{3} = 'NN'; % BB, EST, NN
% approach_string{3} = 'BB';

K = 1; % Number of timelines
N = 3;


%% Setup Supervised Learning Function

mode_stack = 'LIFO';
RP = 0.040; % Resourse Period in ms
Tmax = 50; % Maximum time of simulation in secondes

%% Generate Search Tasks
SearchParams.NbeamsPerRow = [28 29 14 9 10 9 8 7 6];
% SearchParams.NbeamsPerRow = [208 29 14 9 10 9 8 7 6]; % Overload

SearchParams.DwellTime = [36 36 36 18 18 18 18 18 18]*1e-3;
SearchParams.RevistRate = [2.5 5 5 5 5 5 5 5 5]; 
SearchParams.RevisitRateUB = SearchParams.RevistRate + 0.1; % Upper Bound on Revisit Rate
SearchParams.Penalty = 100*ones(size(SearchParams.RevistRate)); % Penalty for exceeding UB
SearchParams.Slope = 1./SearchParams.RevistRate;
Nsearch = sum(SearchParams.NbeamsPerRow);
SearchParams.JobDuration = [];
SearchParams.JobSlope = [];
SearchParams.DropTime = [];
SearchParams.DropCost = [];
for jj = 1:length(SearchParams.NbeamsPerRow)
    SearchParams.JobDuration = [ SearchParams.JobDuration  ; repmat( SearchParams.DwellTime(jj), SearchParams.NbeamsPerRow(jj), 1)];
    SearchParams.JobSlope = [ SearchParams.JobSlope  ; repmat( SearchParams.Slope(jj), SearchParams.NbeamsPerRow(jj), 1)];
    SearchParams.DropTime = [ SearchParams.DropTime; repmat(  SearchParams.RevisitRateUB(jj), SearchParams.NbeamsPerRow(jj), 1)];
    SearchParams.DropCost = [ SearchParams.DropCost; repmat(  SearchParams.Penalty(jj), SearchParams.NbeamsPerRow(jj), 1)];
end


%% Generate Track Tasks
Ntrack = 10;

% Spawn tracks with uniformly distributed ranges and velocity
MaxRangeNmi = 200; %
MaxRangeRateMps = 343; % Mach 1 in Mps is 343


truth.rangeNmi = MaxRangeNmi*rand(Ntrack,1);
truth.rangeRateMps = 2*MaxRangeRateMps*rand(Ntrack,1) - MaxRangeRateMps ;

TrackParams.DwellTime = [18 18 18]*1e-3;
TrackParams.RevisitRate = [1 2 4];
TrackParams.RevisitRateUB = TrackParams.RevisitRate  + 0.1;
TrackParams.Penalty = 300*ones(size(TrackParams.DwellTime));
TrackParams.Slope = 1./TrackParams.RevisitRate;
TrackParams.JobDuration = [];
TrackParams.JobSlope = [];
TrackParams.DropTime = [];
TrackParams.DropCost = [];
for jj = 1:Ntrack
    if  truth.rangeNmi(jj) <= 50
        TrackParams.JobDuration = [TrackParams.JobDuration; TrackParams.DwellTime(1)   ];
        TrackParams.JobSlope = [TrackParams.JobSlope;  TrackParams.Slope(1) ];
        TrackParams.DropTime = [ TrackParams.DropTime;   TrackParams.RevisitRateUB(1)];
        TrackParams.DropCost = [ TrackParams.DropCost;   TrackParams.Penalty(1) ];
    elseif truth.rangeNmi(jj) > 50 &&  abs(truth.rangeRateMps(jj)) >= 100
        TrackParams.JobDuration = [TrackParams.JobDuration; TrackParams.DwellTime(2)   ];
        TrackParams.JobSlope = [TrackParams.JobSlope;  TrackParams.Slope(2) ];
        TrackParams.DropTime = [ TrackParams.DropTime;   TrackParams.RevisitRateUB(2)];
        TrackParams.DropCost = [ TrackParams.DropCost;   TrackParams.Penalty(2) ];
    else
        TrackParams.JobDuration = [TrackParams.JobDuration; TrackParams.DwellTime(3)   ];
        TrackParams.JobSlope = [TrackParams.JobSlope;  TrackParams.Slope(3) ];
        TrackParams.DropTime = [ TrackParams.DropTime;   TrackParams.RevisitRateUB(3)];
        TrackParams.DropCost = [ TrackParams.DropCost;   TrackParams.Penalty(3) ];
    end  
end



% track.duration = 18e-3; % 5 ms (maybe 9 ms)
% t_drop_track = zeros(Ntrack,1); 
% 
% % Create Tiered Revisit rates
% % Tier 1 anything close by
% tier_RR = [0.5 1 4];
% % tier_RR = [RP*1,RP*2,RP*4];
% t_drop_track( truth.rangeNmi <= 50 ) = tier_RR(1); % 1 second revisit rate
% % Tier 2 far away and fast
% t_drop_track( truth.rangeNmi > 50 &  abs(truth.rangeRateMps) >= 100  ) = tier_RR(2); % 1 second revisit rate
% % Tier 3 far away and slow
% t_drop_track( truth.rangeNmi > 50 &  abs(truth.rangeRateMps) < 100  ) = tier_RR(3); % 1 second revisit rate
% w_track = 1./t_drop_track;

plot_en = 1;
if plot_en
    figure(1); clf; hold all; grid on;
    tt = 0:0.1:6;
    plot(tt,costPWlinear(tt, SearchParams.Slope' ,0, SearchParams.RevisitRateUB', SearchParams.Penalty'),'LineWidth',3)
    plot(tt,costPWlinear(tt, TrackParams.Slope' ,0, TrackParams.RevisitRateUB', TrackParams.Penalty'),'LineWidth',3)

    %     plot(tt,cost_linear(tt, 1/tier_RR(1), 0))
%     plot(tt,cost_linear(tt, 1/tier_RR(2), 0))
%     plot(tt,cost_linear(tt, 1/tier_RR(3), 0))
%     legend('Search','Track 1','Track 2', 'Track 3','Location','best')
    xlabel('Time (s)')
    ylabel('Cost')
    title('Cost vs. Time')
    pretty_plot(gcf)
end



loss_mc = zeros(Tmax/(RP/K),length(approach_string));
t_run_mc = zeros(Tmax/(RP/K),length(approach_string));

TaskSequence = zeros(N,Tmax/(RP/K),length(approach_string));
TaskExecution = zeros(N,Tmax/(RP/K),length(approach_string));
ChannelRecord = zeros(K,Tmax/(RP/K),length(approach_string));

% Load Required Neural Network
net = [];
NNstring = sprintf('./NN_REPO/net_task_%i_K_%i_FINAL.mat',N,K);
if any(strcmpi(approach_string,'NN_single'))
    load(NNstring)
elseif any(strcmpi( approach_string ,'NN_Multiple'))
    load('./NN_REPO/net_task_8_K_2_FINAL.mat')
elseif any(strcmpi(approach_string,'MCTS'))
    load(NNstring)  
elseif any(strcmpi(approach_string,'NN'))
%     NNstring = './NN_REPO/net_task_4_K_1_Filter16_20200409T163534.mat';   
    NNstring = './NN_REPO/net_task_4_K_1_Filter16_20200409T192710.mat';
    load(NNstring)
end

data.net = net;


for IterAlg = 1:length(approach_string)
    
    RECORD(IterAlg).s_task = [];
    RECORD(IterAlg).w_task = [];
    RECORD(IterAlg).deadline_task = [];
    RECORD(IterAlg).length_task = [];
    RECORD(IterAlg).drop_task = [];
    RECORD(IterAlg).ChannelAvailableTime = [];
    RECORD(IterAlg).timeSec = [];
    
    %% Generate Data to be scheduled in each dwell
    
    % Initialize master stack
    % stack=java.util.Stack();
    stack = Rstack();
    job = struct('Id',0,'slope',[],'StartTime',0,'DropTime',[],'DropRelativeTime',[],'DropCost',0,'Duration',0,'Type',[],'Priority',0); % Place Holder for Job Description
    job_master = job;
    
    cnt = 1;
    for jj = 1:Nsearch
        job.Id = cnt;
        job.slope = SearchParams.JobSlope(jj);
        job.StartTime = 0;
        job.DropTime = SearchParams.DropTime(jj);
        job.DropRelativeTime = SearchParams.DropTime(jj) + job.StartTime; % Drop time relative to start time
        job.DropCost = SearchParams.DropCost(jj);
        %     job.DropTime = t_drop_search;
        %     job.DropCost = c_drop_search;
        job.Duration = SearchParams.JobDuration(jj);
        if job.slope == 0.4 %Horizon Search
            job.Type = 'HS';
        else % Above horizon search (AHS)
            job.Type = 'AHS';
        end
        job.Priority = costPWlinear(0,job.slope,job.StartTime,job.DropRelativeTime,job.DropCost); % Initially clock is 0
        stack.push(job);
        job_master(cnt) = job; cnt = cnt + 1;
    end
    
    LastSearchId = cnt-1; % Used to find surviellance frame times
    
    for jj = 1:Ntrack
        job.Id = cnt;
        job.slope = TrackParams.JobSlope(jj);%w_track(jj);
        job.StartTime = 0;
        job.DropTime = TrackParams.DropTime(jj);
        job.DropRelativeTime = TrackParams.DropTime(jj) + job.StartTime; % Drop time relative to start time
        job.DropCost = TrackParams.DropCost(jj);
        %     job.DropTime = t_drop_track(jj);
        %     job.DropCost = c_drop_search;
        job.Duration = TrackParams.JobDuration(jj);%track.duration;   
        if job.slope == 0.25
            job.Type = 'TLow';
        elseif job.slope == 0.5
            job.Type = 'TMed';
        else
            job.Type = 'THigh';
        end
%         TrackIndex = find(w_track(jj) == unique(w_track));
%         job.Type = ['T' num2str(TrackIndex)];
        job.Priority = costPWlinear(0,job.slope,job.StartTime,job.DropRelativeTime,job.DropCost); % Initially clock is 0

%         job.Priority = cost_linear(0,job.slope,job.StartTime); % Initially clock is 0
        stack.push(job);
        job_master(cnt) = job; cnt = cnt + 1;
    end
    
    
    
%     Capacity = sum([job_master.slope].*[job_master.Duration]);
    Capacity = sum([job_master.slope].*round([job_master.Duration]/(RP/2))*RP/2);
    
    fprintf('Timeline Capacity: %f \n\n',Capacity)

    
    %% Begin Simulation Loop
    % Specify number of task to process at any given time
%     N = RP/search.duration;
%     N = 8;
    N_mc = 1;
    i_mc = 1; % Used for Monte Carlo index. set to 1 initially later add loop
    
%     N_alg = numel(fcn_search);
    N_alg = 1;
    
    
    X = [];
    Y = [];
    
%     JobRevistTime = cell(size(job_master,2),1);
    metrics.JobRevistCount = zeros(size(job_master,2),1);
    metrics.JobType = {job_master.Type};
 
    
    tstart = tic;
    
    iter = 1;
    ChannelAvailableTime = zeros(K,1);
    for timeSec = 0:RP/K:Tmax
        timeSec = round( timeSec/(RP/K))*(RP/K);
        
        if min(ChannelAvailableTime) > timeSec % Don't schedule unless a channel is free
            continue
        end
        
        if mod(timeSec,RP*10) == 0
            fprintf('Time = %0.2f \n', timeSec)
        end
        
        % Reassess Track Priorities ( Need to reshuffle jobs based on current cost
        % of each delayed task )
        for n = 1:size(job_master,2)
            job_master(n).Priority = costPWlinear(timeSec,job_master(n).slope,job_master(n).StartTime,job_master(n).DropRelativeTime, job_master(n).DropCost  );
            if job_master(n).Priority == Inf
                job_master(n).Priority = -Inf; % Reassign to make lower priority
            end
            if job_master(n).Priority > 10
%                 keyboard
            end
        end        
              
        
        
        if sum([job_master.Priority] ~= -Inf) < N 
%             keyboard
            continue
        end
            
%         figure(111); clf; hold all;
%         plot([job_master.StartTime],'--x')
%         plot([job_master.Priority],'-o')
%         grid on
%         title(sprintf('Current Time = %f',timeSec))
%         pause
        

        if FLAG.FixedPriority == 1
            priorityIdx = [1:length(job_master)];
        else        
            [~,priorityIdx] = sort([job_master.Priority],'descend');
        end
        job_master = job_master(priorityIdx);
       
        fprintf('Iteration %i \n',iter)
        T = struct2table(job_master);
        if mod(timeSec,RP*10) == 0
            disp(T)
        end
        
        % Initially all task have same start time Take first Ntasks to schedule
        queue = job_master(1:N);
        job_master(1:N) = []; % Remove jobs being scheduled
        %     w = [queue.slope];
        s_task      = [queue.StartTime]';
        length_task = [queue.Duration]';
        w_task      = [queue.slope]';
        drop_task   = [queue.DropCost]';
        deadline_task = [queue.DropRelativeTime]';
        %     t_drop = [queue.DropTime;
        
        
   
        
        
        queueID(:,iter,IterAlg)= [queue.Id];
        queueRecord{iter,IterAlg} = queue;
        
        %     metrics.JobRevistTime( [queue.Id] ,metrics.JobRevistCount([queue.Id]) ) = timeSec;
        
        
        % Anonymous functions can be slowwwwww ... probably can vectorize the call
        % to this function to speed things up
        l_task = cell(N,1);
        for n = 1:N
            l_task{n} = @(t) cost_linDrop(t, queue(n).slope ,  queue(n).StartTime  ,  queue(n).DropTime  ,  queue(n).DropCost );
        end
        
        % Schedule Tasks using BB and generate relevant sampled data
%         drop_task = zeros(N,1); deadline_task = 100*ones(N,1);
        
        
        data.N = N;
        data.K = K;
        data.s_task = s_task;
        data.w_task = w_task;
        data.deadline_task = deadline_task; %(deadline_task + s_task); % Updated so it's relative to release time already 09APR2020
        data.length_task = length_task;
        data.drop_task = drop_task;
        data.RP = RP;
        data.ChannelAvailableTime = ChannelAvailableTime;
        data.scheduler = 'flexdar';
        data.timeSec = timeSec;
        
        RECORD(IterAlg).s_task = [RECORD(IterAlg).s_task s_task];
        RECORD(IterAlg).w_task = [RECORD(IterAlg).w_task w_task];
        RECORD(IterAlg).deadline_task = [RECORD(IterAlg).deadline_task deadline_task];
        RECORD(IterAlg).length_task = [RECORD(IterAlg).length_task length_task];
        RECORD(IterAlg).drop_task = [RECORD(IterAlg).drop_task drop_task];
        RECORD(IterAlg).ChannelAvailableTime = [RECORD(IterAlg).ChannelAvailableTime ChannelAvailableTime];
        RECORD(IterAlg).timeSec = [RECORD(IterAlg).timeSec timeSec];
%         RECORD.drop_task = [record.drop_task drop_task];
%         RECORD.drop_task = [record.drop_task drop_task];

        
        for i_a = 1:N_alg
                       
            [loss,t_run,T,t_ex,ChannelAvailableTime] = PerformTaskAssignment(approach_string,IterAlg,data);
                        
            %         [t_ex,loss,t_run,Xnow,Ynow] = fcn_search{i_a}(s_task,d_task,l_task,timeSec);
            
            loss_mc(iter,IterAlg) = loss;
            t_run_mc(iter,IterAlg) = t_run;
            TaskSequence(:,iter,IterAlg) = T(1:N);
            TaskExecution(:,iter,IterAlg) = t_ex;
            ChannelRecord(:,iter,IterAlg) = ChannelAvailableTime;
            
            if exist('Xnow')
                X = cat(3,X,Xnow);
                Y = [Y; Ynow];
            end
        end
        
        
        job_type = [queue.Type];
        occupancy.search(iter) = sum(job_type == 'S')/N;
        occupancy.track(iter) = sum(job_type == 'T')/N;
        
%         [~,sortIdx] = sort(t_ex); Don't think this is needed anymore
%         10APR2020
        
        new_job = struct('Id',0,'slope',[],'StartTime',0,'DropTime',[],'DropRelativeTime',[],'DropCost',0,'Duration',0,'Type',[],'Priority',0); % Place Holder for Job Description
        
        % Only Execute Jobs that need to be
        
        indexExecution = find( (t_ex + 1e-8) < (timeSec + RP) )'; % Get a numerical error that messes this up
        indexNoExecution = setdiff([1:N],indexExecution); % Don't update these jobs
      
        for n = indexExecution
            new_job(n).Id = queue((n)).Id;
            new_job(n).StartTime = t_ex((n)) + queue((n)).Duration ;
            new_job(n).slope = queue((n)).slope;
            new_job(n).DropTime = queue((n)).DropTime;
            new_job(n).DropRelativeTime = queue((n)).DropTime + new_job(n).StartTime; % Update with new start time and job DropTime           
            new_job(n).DropCost = queue((n)).DropCost;
            new_job(n).Duration = queue((n)).Duration;
            new_job(n).Type = queue((n)).Type;
            metrics.JobRevistCount([queue(n).Id]) = metrics.JobRevistCount([queue(n).Id]) + 1;
            JobRevistTime{ queue(n).Id }( metrics.JobRevistCount(queue(n).Id) )     = timeSec;          
        end
        % Update Channel Available Time
        if length(indexExecution) > 0
            ChannelAvailableTime = new_job(n).StartTime; % Use last visited index n
        else
            keyboard
        end
        
        
        for n = indexNoExecution
            new_job(n).Id = queue((n)).Id;
            new_job(n).StartTime = queue((n)).StartTime;
            new_job(n).slope = queue((n)).slope;
            new_job(n).DropTime = queue((n)).DropTime;
            new_job(n).DropRelativeTime = queue((n)).DropRelativeTime; % Update with new start time and job DropTime           
            new_job(n).DropCost = queue((n)).DropCost;
            new_job(n).Duration = queue((n)).Duration;
            new_job(n).Type = queue((n)).Type;
        end
      
        
        
        job_master = [job_master, new_job];
        
        %     formatJobsFcn(job_master)
        
        %     disp( [job_master.StartTime] )
        %     disp({job_master.Type})
        
        % Update Track Truth Positions
        pos = truth.rangeNmi * 1852;
        vel = truth.rangeRateMps;
        truth.rangeNmi = ( pos + (timeSec + RP)*vel ) /1852;
        
        
        iter = iter + 1;        
    end
    loss_mc(iter:end,:) = []; % Remove extra entries
    t_run_mc(iter:end,:) = [];
    
    TimeElapsed(IterAlg) = toc(tstart);
    
    fprintf('Elapsed Time %f \n\n',TimeElapsed(IterAlg))
    
    
    %% Diagnostics    
    for n = 1:length(metrics.JobRevistCount)
        try 
            metrics.RevisitRate(n) =  mean( diff([JobRevistTime{n}] ));
        catch
            metrics.RevisitRate(n) = 0;
        end
    end
    

    metrics.UniqueJobTypes = unique(metrics.JobType);       
    for jj = 1:length(metrics.UniqueJobTypes)
       JobIndex = find( strcmpi(metrics.JobType,metrics.UniqueJobTypes(jj)) );
       metrics.JobTypeRR(jj) = mean(metrics.RevisitRate(JobIndex));
    end
    
    
%     LastSearchId = min(LastSearchId,length(JobRevistTime));
    SurvFrameTime = JobRevistTime{LastSearchId};
    AvgSurvFrameTime = mean(diff(SurvFrameTime));
    
    desiredRevisitRate = 1./[job_master.slope];
    desiredRevisitRate([job_master.Id]) = desiredRevisitRate; % Sort by Id number 1:NumIds
    
    RawUtility = desiredRevisitRate - metrics.RevisitRate;
    RawPenalty  = RawUtility;
    RawPenalty(RawPenalty > 0) = 0; % Pass/Fail anything that's positive ignore
    TotalUtility = sum(RawUtility); % More positive is better
    TotalPenalty = sum(RawPenalty);    % Less negative is better
    penalty_vec(IterAlg) = TotalPenalty;
    
    
    fprintf('Total Penalty %f \n\n',TotalPenalty)
    
    
    
    
%     figure(2 + (IterAlg-1)*4); clf;
    figure(2);
    subplot(length(approach_string),1,IterAlg)

    % subplot(2,2,1)
    hold all; grid on;
    plot(occupancy.search)
    plot(occupancy.track)
    legend('Search','Track')
    xlabel('Iteration')
    ylabel('Occupancy')
    title(['Job Occupancy: ' approach_string{IterAlg}])
    pretty_plot(gcf)
    fname = ['.\Figures\' approach_string{IterAlg} '_Job_Occupancy'];
    if FLAG.save
        saveas(gcf,[fname '.fig'])
        saveas(gcf,[fname '.epsc'])
    end
    
    figure(3 + (IterAlg-1)*4); clf;
    % subplot(2,2,2)
    hold all; grid on;
    for n = 1:size(JobRevistTime,2)
        plot( JobRevistTime{n} , ones(size(JobRevistTime{n})) + (n-1) ,'x' )
    end
    xlabel('Revist Time (s)')
    ylabel('Job Id')
    title(['Job Revisit Time: ' approach_string{IterAlg} ])
    pretty_plot(gcf)
    if FLAG.save
        fname = ['.\Figures\' approach_string{IterAlg} '_Job_Revisit_Time'];
        saveas(gcf,[fname '.fig'])
        saveas(gcf,[fname '.epsc'])
    end
    
%     figure(4 + (IterAlg-1)*4); clf;
    figure(4); 
    subplot(length(approach_string),1,IterAlg)
    % subplot(2,2,[3]);
    cla; hold all;
    plot(metrics.RevisitRate,[1:size(metrics.RevisitRate,2)],'bd','MarkerSize',8,'LineWidth',3)
    plot(1./[job_master.slope],[job_master.Id],'ro')
    
    xlabel('Revist Rate (s)')
    ylabel('Job Id')
    grid on;
    aa = axis;
    xlim([0 aa(2)]);
    xpos = aa(2)*.25;
    ypos = (aa(4) - aa(3))*.25 + aa(3);
%     text(AvgSurvFrameTime,LastSearchId,['Avg. Surv. Frame Time = ' num2str(AvgSurvFrameTime) '\rightarrow'],'LineWidth',6,'HorizontalAlignment','right')
    for nn = 1:length(metrics.UniqueJobTypes) 
        SearchId(nn) = find( strcmp( metrics.JobType,   metrics.UniqueJobTypes{nn}),1,'last');      
        text(metrics.JobTypeRR(nn), SearchId(nn)  ,[metrics.UniqueJobTypes{nn} ' = '  num2str(metrics.JobTypeRR(nn)) '\rightarrow'],'LineWidth',6,'HorizontalAlignment','right')
    end
    
    title(sprintf('Job Revisit Rate %s \n Utility = %0.2f,  Penalty = %0.2f',approach_string{IterAlg},TotalUtility,TotalPenalty))
    legend('Achieved Rate','Desired Rate')
    pretty_plot(gcf)
    if FLAG.save
        fname = ['.\Figures\' approach_string{IterAlg} '_Achieved_Rate'];
        saveas(gcf,[fname '.fig'])
        saveas(gcf,[fname '.epsc'])
    end
    
    if IterAlg == 1
        figure(5 + (IterAlg-1)*4); clf;
        % subplot(2,2,4);
        cla;
        plot([job_master.slope],[job_master.Id],'o')
        xlabel('Cost Slope')
        ylabel('Job Id')
        title(['Final Job Priority: ' approach_string{IterAlg}])
        pretty_plot(gcf)
        if FLAG.save
            fname = ['.\Figures\' approach_string{IterAlg} '_Job_Priority'];
            saveas(gcf,[fname '.fig'])
            saveas(gcf,[fname '.epsc'])
        end
    end
end

%% Final Plots
% leg_str{1} = 'EST';
% leg_str{2} = 'NN';
% leg_str{3} = 'BB';
% penalty_vec = [-0.128465 -0.030758 -0.015145];
leg_str = approach_string;
shape = 'oxsd';
time_vec = TimeElapsed./(iter-1)*1000;
% time_vec = [0.547488 3.0350  27.202844]/51*1000;

figure(106); clf;
hold all; grid on
for jj = 1:length(approach_string)
    plot(time_vec(jj),-penalty_vec(jj),shape(jj),'MarkerSize',10,'LineWidth',3)
end
% plot(time_vec(2),-penalty_vec(2),'x','MarkerSize',10,'LineWidth',3)
% plot(time_vec(3),-penalty_vec(3),'s','MarkerSize',10,'LineWidth',3)
legend(leg_str,'Location','best')
ylabel('Cumulative Penalty')
xlabel('Computation Time (ms)')
title({'Radar Performance Metric', 'Computation Time vs. Penalty (Closer to 0 \rightarrow Better Performance)'})

pretty_plot(gcf)
if FLAG.save
    fname = ['.\Figures\' 'Compute_Time'];
    saveas(gcf,[fname '.fig'])
    saveas(gcf,[fname '.epsc'])
end


figure(107); clf;
AvgCost = mean(loss_mc);
% for jj = 1:size(loss_mc,2)
%     AvgCost(jj) = mean( loss_mc( loss_mc(:,jj) > 0,jj));
% end
AvgTime = mean(t_run_mc)*1000;
clf;
hold all; grid on
for jj = 1:length(approach_string)
    plot(AvgTime(jj),AvgCost(jj),shape(jj),'MarkerSize',10,'LineWidth',3)
end
% plot(time_vec(2),-penalty_vec(2),'x','MarkerSize',10,'LineWidth',3)
% plot(time_vec(3),-penalty_vec(3),'s','MarkerSize',10,'LineWidth',3)
legend(leg_str,'Location','best')
ylabel('Cost')
xlabel('Computation Time (ms)')
title({'N Task Scheduling Cost','Computation Time vs. Cost'})

pretty_plot(gcf)
if FLAG.save
    fname = ['.\Figures\' 'Cost_vs_Time'];
    saveas(gcf,[fname '.fig'])
    saveas(gcf,[fname '.epsc'])
end

figure(108); clf;
b = ones(10,1)/10;
for IterAlg = 1:length(approach_string)
    A(:,IterAlg) = conv(loss_mc(:,IterAlg),b);
end
plot(A); grid on;
xlabel('Iteration')
ylabel('Smoothed Loss')
title('Smoothed Scheduling Loss vs. Iteration')
legend(approach_string)
pretty_plot(gcf)

%% Diagnostic plots
DIAG = 0; 
if DIAG == 1
    figure(109); clf;
    BB = RECORD(2).s_task - min(RECORD(2).s_task);
    hist( BB(:) ,100 )
    title('Distribution of Release Times vs. Minimum Release Time')
    
    figure(110); clf;
    CC = RECORD(2).deadline_task - min(RECORD(2).s_task);
    hist( CC(:) ,100 )
    title('Distribution of Deadlines vs. Minimum Release Time')
    
    figure(111); clf;
    hist( RECORD(2).w_task(1,:),100)
    title('Distribution of Weights')
    
    figure(112); clf;
    hist( RECORD(2).length_task(1,:),100)
    title('Distribution of Task Lengths')

end 

%%

if FLAG.profile
    profile viewer
end
