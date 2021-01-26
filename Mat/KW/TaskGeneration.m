
clearvars
close all
seed = 12307;
rng(seed)

addpath('./functions/')

%% Setup Supervised Learning Function

fcn_search = {@(s_task,d_task,l_task,refTime) fcn_BB_NN(s_task,d_task,l_task,'LIFO',refTime);};


% fcn_BB_NN(s_task,d_task,l_task,'LIFO',refTime);
% fcn_BB_NN(s_task,d_task,l_task,'LIFO',0.04);


RP = 0.040; % Resourse Period in ms
Tmax = 100; % Maximum time of simulation in secondes

%% Generate Search Tasks
Nsearch = 10;
search.duration = 5e-3; % 4.5 ms (maybe 9 ms)

Search_RR = (Nsearch+4)*search.duration; % Desired Search revisit rate


slope_search = 1/Nsearch*search.duration;
search_drop_cost = 10;

t_drop_search = Search_RR;
c_drop_search = max(slope_search*Search_RR,search_drop_cost) ; % Make sure dropping cost is greater than cost reached after waiting for revisit rate
w_search = slope_search;




%% Generate Track Tasks
Ntrack = 6;

% Spawn tracks with uniformly distributed ranges and velocity
MaxRangeNmi = 200; %
MaxRangeRateMps = 343; % Mach 1 in Mps is 343


truth.rangeNmi = MaxRangeNmi*rand(Ntrack,1);
truth.rangeRateMps = 2*MaxRangeRateMps*rand(Ntrack,1) - MaxRangeRateMps ;

track.duration = 5e-3; % 5 ms (maybe 9 ms)
t_drop_track = zeros(Ntrack,1);
w_track = 1;
c_drop_search = 10;

% Create Tiered Revisit rates
% Tier 1 anything close by
t_drop_track( truth.rangeNmi <= 50 ) = 0.5; % 1 second revisit rate

% Tier 2 far away and fast
t_drop_track( truth.rangeNmi > 50 &  abs(truth.rangeRateMps) >= 100  ) = 1; % 1 second revisit rate

% Tier 3 far away and slow
t_drop_track( truth.rangeNmi > 50 &  abs(truth.rangeRateMps) < 100  ) = 4; % 1 second revisit rate


%% Generate Data to be scheduled in each dwell

% Initialize master stack
% stack=java.util.Stack();
stack = Rstack();
job = struct('slope',[],'StartTime',0,'DropTime',[],'DropCost',0,'Duration',0,'Type',[]); % Place Holder for Job Description
job_master = job;

cnt = 1;
for jj = 1:Nsearch
    job.slope = slope_search;
    job.StartTime = 0;
    job.DropTime = t_drop_search;
    job.DropCost = c_drop_search;
    job.Duration = search.duration;
    job.Type = 'S';
    stack.push(job);
    job_master(cnt) = job; cnt = cnt + 1;
end

for jj = 1:Ntrack
    job.slope = w_track;
    job.StartTime = 0;
    job.DropTime = t_drop_track(jj);
    job.DropCost = c_drop_search;
    job.Duration = track.duration;
    job.Type = 'T';
    stack.push(job);
    job_master(cnt) = job; cnt = cnt + 1;
end



%% Begin Simulation Loop
% Specify number of task to process at any given time
N = RP/search.duration;
N_mc = 1;
i_mc = 1; % Used for Monte Carlo index. set to 1 initially later add loop

N_alg = numel(fcn_search);

loss_mc = zeros(N_mc,N_alg);
t_run_mc = zeros(N_mc,N_alg);
X = [];
Y = [];
tstart = tic;


for timeSec = 0:RP:Tmax
    
    
    if mod(timeSec,RP*10) == 0
        fprintf(['Time = ' num2str(timeSec)])
    end
    
    % Initially all task have same start time Take first Ntasks to schedule
    queue = job_master(1:N);
    job_master(1:N) = []; % Remove jobs being scheduled
    %     w = [queue.slope];
    s_task = [queue.StartTime]';
    d_task = [queue.Duration]';
    %     t_drop = [queue.DropTime;
    
    % Anonymous functions can be slowwwwww ... probably can vectorize the call
    % to this function to speed things up
    l_task = cell(N,1);
    for n = 1:N
        l_task{n} = @(t) cost_linDrop(t, queue(n).slope ,  queue(n).StartTime  ,  queue(n).DropTime  ,  queue(n).DropCost );
    end
    
    % Schedule Tasks using BB and generate relevant sampled data
    for i_a = 1:N_alg
        [t_ex,loss,t_run,Xnow,Ynow] = fcn_search{i_a}(s_task,d_task,l_task,timeSec);
        
        loss_mc(i_mc,i_a) = loss;
        t_run_mc(i_mc,i_a) = t_run;
        
        
        X = cat(3,X,Xnow);
        Y = [Y; Ynow];
    end
    
    
    [~,sortIdx] = sort(t_ex);
    
    new_job = struct('slope',[],'StartTime',0,'DropTime',[],'DropCost',0,'Duration',0,'Type',[]); % Place Holder for Job Description
    
    for n = 1:N
        new_job(n).StartTime = t_ex(sortIdx(n)) + queue(sortIdx(n)).Duration ;
        new_job(n).slope = queue(sortIdx(n)).slope;
        new_job(n).DropTime = queue(sortIdx(n)).DropTime;
        new_job(n).DropCost = queue(sortIdx(n)).DropCost;
        new_job(n).Duration = queue(sortIdx(n)).Duration;
        new_job(n).Type = queue(sortIdx(n)).Type;
    end
    
    job_master = [job_master, new_job];
    
    
%     disp( [job_master.StartTime] )
%     disp({job_master.Type})
    
    % Update Track Truth Positions
    pos = truth.rangeNmi * 1852;
    vel = truth.rangeRateMps;
    truth.rangeNmi = ( pos + (timeSec + RP)*vel ) /1852;
    
    
    
    % Reassess Track Priorities
    
    
    
end
