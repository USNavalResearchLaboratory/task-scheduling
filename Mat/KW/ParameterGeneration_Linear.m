clearvars
close all

FLAG.profile = 1;
FLAG.plot = 0;

if FLAG.profile
    profile clear
    profile on -history
end

seed = 12307;
rng(seed)

addpath('./functions/')


%% Setup up MONTE CARLOS
MONTE = 10;
for monte = 1:MONTE
    
    %% Setup Supervised Learning Function
    
    mode_stack = 'LIFO';
    fcn_search = {@(s_task,d_task,l_task,refTime) fcn_BB_NN(s_task,d_task,l_task,mode_stack,refTime);};
    
    
    % fcn_BB_NN(s_task,d_task,l_task,'LIFO',refTime);
    % fcn_BB_NN(s_task,d_task,l_task,'LIFO',0.04);
    
    
    RP = 0.040; % Resourse Period in ms
    Tmax = 2; % Maximum time of simulation in secondes
    
    %% Generate Search Tasks (Same across all monte carlos)
    Nsearch = 40;
    search.duration = 5e-3; % 4.5 ms (maybe 9 ms)
    
    % Search_RR = Nsearch*search.duration;
    Search_RR = 8*RP;
    % Search_RR = (Nsearch+4)*search.duration; % Desired Search revisit rate
    
    slope_search = 1/Search_RR; % Set slope so that cost is equal to 1 at revisit rate
    
    
    
    %% Generate Track Tasks (New tracks every Monte Carlo)
    MaxNumTrack = 20; % Every monte carlo generate a new maximum number of tracks
    Ntrack = randi(MaxNumTrack);
    
    % Spawn tracks with uniformly distributed ranges and velocity
    MaxRangeNmi = 200; %
    MaxRangeRateMps = 343; % Mach 1 in Mps is 343
    
    
    truth.rangeNmi = MaxRangeNmi*rand(Ntrack,1);
    truth.rangeRateMps = 2*MaxRangeRateMps*rand(Ntrack,1) - MaxRangeRateMps ;
    
    track.duration = 5e-3; % 5 ms (maybe 9 ms)
    t_drop_track = zeros(Ntrack,1);
    % w_track = 1;
    % c_drop_search = 10;
    
    % Create Tiered Revisit rates
    % Tier 1 anything close by
    tier_RR = [0.5 1 4];
    tier_RR = [RP*1,RP*2,RP*4];
    
    t_drop_track( truth.rangeNmi <= 50 ) = tier_RR(1); % 1 second revisit rate
    
    % Tier 2 far away and fast
    t_drop_track( truth.rangeNmi > 50 &  abs(truth.rangeRateMps) >= 100  ) = tier_RR(2); % 1 second revisit rate
    
    % Tier 3 far away and slow
    t_drop_track( truth.rangeNmi > 50 &  abs(truth.rangeRateMps) < 100  ) = tier_RR(3); % 1 second revisit rate
    
    w_track = 1./t_drop_track;
    
    plot_en = 1;
    if plot_en
        figure(1); clf; hold all; grid on;
        tt = 0:01:3;
        plot(tt,cost_linear(tt, slope_search ,0))
        plot(tt,cost_linear(tt, 1/tier_RR(1), 0))
        plot(tt,cost_linear(tt, 1/tier_RR(2), 0))
        plot(tt,cost_linear(tt, 1/tier_RR(3), 0))
        legend('Search','Track 1','Track 2', 'Track 3','Location','best')
    end
    
    %% Generate Data to be scheduled in each dwell
    
    % Initialize master stack
    % stack=java.util.Stack();
    stack = Rstack();
    job = struct('Id',0,'slope',[],'StartTime',0,'DropTime',[],'DropCost',0,'Duration',0,'Type',[],'Priority',0); % Place Holder for Job Description
    job_master = job;
    
    cnt = 1;
    for jj = 1:Nsearch
        job.Id = cnt;
        job.slope = slope_search;
        job.StartTime = 0;
        %     job.DropTime = t_drop_search;
        %     job.DropCost = c_drop_search;
        job.Duration = search.duration;
        job.Type = 'S';
        job.Priority = cost_linear(0,slope_search,job.StartTime); % Initially clock is 0
        stack.push(job);
        job_master(cnt) = job; cnt = cnt + 1;
    end
    
    LastSearchId = cnt-1; % Used to find surviellance frame times
    
    for jj = 1:Ntrack
        job.Id = cnt;
        job.slope = w_track(jj);
        job.StartTime = 0;
        %     job.DropTime = t_drop_track(jj);
        %     job.DropCost = c_drop_search;
        job.Duration = track.duration;
        job.Type = 'T';
        job.Priority = cost_linear(0,slope_search,job.StartTime); % Initially clock is 0
        stack.push(job);
        job_master(cnt) = job; cnt = cnt + 1;
    end
    
    
    
    %% Begin Simulation Loop
    % Specify number of task to process at any given time
    N = RP/search.duration;
    % N = 8;
    N_mc = 1;
    i_mc = 1; % Used for Monte Carlo index. set to 1 initially later add loop
    
    N_alg = numel(fcn_search);
    
    loss_mc = zeros(N_mc,N_alg);
    t_run_mc = zeros(N_mc,N_alg);
    X = [];
    Y = [];
    
    
    metrics.JobRevistCount = zeros(size(job_master,2),1);
    
    tstart = tic;
    
    iter = 1;
    if monte == 1
        parameters.s_task = zeros(N,round(Tmax/RP),MONTE );
        parameters.w_task = zeros(N,round(Tmax/RP),MONTE );
    end
    for timeSec = 0:RP:Tmax-RP
        
        
        if mod(timeSec,RP*10) == 0
            fprintf('Time = %0.2f \n', timeSec)
        end
        
        % Reassess Track Priorities ( Need to reshuffle jobs based on current cost
        % of each delayed task )
        for n = 1:size(job_master,2)
            job_master(n).Priority = cost_linear(timeSec,job_master(n).slope,job_master(n).StartTime);
        end
        
        
        [~,priorityIdx] = sort([job_master.Priority],'descend');
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
        s_task = [queue.StartTime]';
        d_task = [queue.Duration]';
        w_task = [queue.slope]';
        %     t_drop = [queue.DropTime;
        
        
        metrics.JobRevistCount([queue.Id]) = metrics.JobRevistCount([queue.Id]) + 1;
        for n = 1:N
            JobRevistTime{ queue(n).Id }( metrics.JobRevistCount(queue(n).Id) )     = timeSec;
        end
        
        
        
        
        %     metrics.JobRevistTime( [queue.Id] ,metrics.JobRevistCount([queue.Id]) ) = timeSec;
        
        
        % Anonymous functions can be slowwwwww ... probably can vectorize the call
        % to this function to speed things up
        l_task = cell(N,1);
        for n = 1:N
            l_task{n} = @(t) cost_linDrop(t, queue(n).slope ,  queue(n).StartTime  ,  queue(n).DropTime  ,  queue(n).DropCost );
        end
        
        % Schedule Tasks using BB and generate relevant sampled data
        for i_a = 1:N_alg
            
            
            [t_ex,loss,t_run] = fcn_ES_linear(s_task,d_task,w_task,timeSec);
            %         [t_ex,loss,t_run,Xnow,Ynow] = fcn_BB_NN_linear(s_task,d_task,w_task,mode_stack,timeSec);
            
            %         [t_ex,loss,t_run,Xnow,Ynow] = fcn_search{i_a}(s_task,d_task,l_task,timeSec);
            
            parameters.s_task(:,iter,monte) = max( s_task - timeSec , 0 );
            parameters.w_task(:,iter,monte) = w_task;
            
            
            loss_mc(i_mc,i_a) = loss;
            t_run_mc(i_mc,i_a) = t_run;
            
            if exist('Xnow')
                X = cat(3,X,Xnow);
                Y = [Y; Ynow];
            end
        end
        
        
        job_type = [queue.Type];
        occupancy.search(iter) = sum(job_type == 'S')/N;
        occupancy.track(iter) = sum(job_type == 'T')/N;
        
        [~,sortIdx] = sort(t_ex);
        
        new_job = struct('Id',0,'slope',[],'StartTime',0,'DropTime',[],'DropCost',0,'Duration',0,'Type',[],'Priority',0); % Place Holder for Job Description
        
        for n = 1:N
            new_job(n).Id = queue(sortIdx(n)).Id;
            new_job(n).StartTime = t_ex(sortIdx(n)) + queue(sortIdx(n)).Duration ;
            new_job(n).slope = queue(sortIdx(n)).slope;
            new_job(n).DropTime = queue(sortIdx(n)).DropTime;
            new_job(n).DropCost = queue(sortIdx(n)).DropCost;
            new_job(n).Duration = queue(sortIdx(n)).Duration;
            new_job(n).Type = queue(sortIdx(n)).Type;
        end
        %     for n = 1:N
        %        new_job(n).Priority = cost_linear(timeSec,new_job(n).slope,new_job(n).StartTime);
        %     end
        
        
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
    
    %% Diagnostics
    for n = 1:size(JobRevistTime,2)
        metrics.RevisitRate(n) =  mean( diff(JobRevistTime{n} ));
    end
    SurvFrameTime = JobRevistTime{LastSearchId};
    AvgSurvFrameTime = mean(diff(SurvFrameTime));
    
    desiredRevisitRate = 1./[job_master.slope];
    desiredRevisitRate([job_master.Id]) = desiredRevisitRate; % Sort by Id number 1:NumIds
    
    RawUtility = desiredRevisitRate - metrics.RevisitRate;
    RawPenalty  = RawUtility;
    RawPenalty(RawPenalty > 0) = 0; % Pass/Fail anything that's positive ignore
    TotalUtility = sum(RawUtility); % More positive is better
    TotalPenalty = sum(RawPenalty);    % Less negative is better
    
    
    if FLAG.plot
        
        
        figure(2+monte); clf;
        subplot(2,2,1)
        hold all; grid on;
        plot(occupancy.search)
        plot(occupancy.track)
        legend('Search','Track')
        xlabel('Iteration')
        ylabel('Occupancy')
        title('Job Occupancy')
        
        % figure(3); clf;
        subplot(2,2,2)
        hold all; grid on;
        for n = 1:size(JobRevistTime,2)
            plot( JobRevistTime{n} , ones(size(JobRevistTime{n})) + (n-1) ,'x' )
        end
        xlabel('Revist Time (s)')
        ylabel('Job Id')
        title('Job Revisit Time')
        
        
        
        subplot(2,2,[3]); cla; hold all;
        plot(metrics.RevisitRate,[1:size(JobRevistTime,2)],'bd','MarkerSize',8,'LineWidth',3)
        plot(1./[job_master.slope],[job_master.Id],'ro')
        
        xlabel('Revist Rate (s)')
        ylabel('Job Id')
        grid on;
        aa = axis;
        xlim([0 aa(2)]);
        xpos = aa(2)*.25;
        ypos = (aa(4) - aa(3))*.25 + aa(3);
        text(AvgSurvFrameTime,LastSearchId,['Avg. Surv. Frame Time = ' num2str(AvgSurvFrameTime) '\rightarrow'],'LineWidth',6,'HorizontalAlignment','right')
        title(sprintf('Job Revisit Rate \n Utility = %0.2f,  Penalty = %0.2f',TotalUtility,TotalPenalty))
        legend('Achieved Rate','Desired Rate')
        
        subplot(2,2,4); cla;
        plot([job_master.slope],[job_master.Id],'o')
        xlabel('Cost Slope')
        ylabel('Job Id')
        title(sprintf('Final Job Priority, No. Track = %i, No. Search = %i',Ntrack  , Nsearch ) )
        pretty_plot(gcf)
        
    end
    
    
    clear JobRevistTime metrics
    
    
end

%%

if FLAG.profile
    profile viewer
end


figure(1001); clf;
[count,edges] = histcounts(parameters.w_task(:),100,'Normalization','probability');
plot(edges(1:end-1) + abs(diff(edges(1:2))),count)
grid on

figure(1002); clf;
hist(parameters.w_task(:),100)
grid on
