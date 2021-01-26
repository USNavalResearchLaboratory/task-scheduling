
%% Algorithms Implemented here are based on "Task Selection and Scheduling in Multifunction Multichannel Radars" by M. Shaghaghi and R. Adve, 978-1-4673-8823-8/17/$31.00 ©2017 IEEE
clearvars


FLAG.profile = 1;

if FLAG.profile
    profile clear
    profile on -history
end

seed = 111;
rng(seed)

MONTE = 100;

N = 16; % Number of Tasks
K = 4; % Number of identical channels or machines
T = 100; % Time window of all tasks


% <<<<<<< HEAD
N_vec = 10:5:20;
% =======
N_vec = 10:5:20;
% N_vec = 16;

% >>>>>>> 7820678eed649ed87b79d4e8f080ff9acef3f601
NumN = length(N_vec);

RunTime.EST = zeros(MONTE,NumN);
RunTime.EstSwap = zeros(MONTE,NumN);
RunTime.ED = zeros(MONTE,NumN);
RunTime.EST = zeros(MONTE,NumN);
RunTime.BB = zeros(MONTE,NumN);

Cost.EST = zeros(MONTE,NumN);
Cost.EstSwap = zeros(MONTE,NumN);
Cost.ED = zeros(MONTE,NumN);
Cost.EdSwap = zeros(MONTE,NumN);
Cost.BB = zeros(MONTE,NumN);


DropPercent.EST = zeros(MONTE,NumN);
DropPercent.EstSwap = zeros(MONTE,NumN);
DropPercent.ED = zeros(MONTE,NumN);
DropPercent.EdSwap = zeros(MONTE,NumN);
DropPercent.BB = zeros(MONTE,NumN);


cnt.N = 1;
for N = N_vec
    for monte = 1:MONTE
        
        s_task = rand(N,1)*100; % Starting time of tasks
        viable_task = 10*rand(N,1) + 2; % Difference between deadline d_n and starting time s_n
        deadline_task = viable_task + s_task; % Task deadline equal to the starting time + viablity window of each task
        length_task = rand(N,1)*9 + 2;  % Task processing length
        drop_task = rand(N,1)*400 + 100; % Dropping cost of each task
        w_task = rand(N,1)*4 + 1;   % Tardiness penalty of each task
        
        
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
        [T,Tscheduled,Tdrop] = BBschedulerStack(K,s_task,deadline_task,length_task,drop_task,w_task);

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
        
        
    end
    cnt.N = cnt.N + 1;
end



%% Graphics
leg_str{1} = 'EST';
leg_str{2} = 'EST Swap';
leg_str{3} = 'ED';
leg_str{4} = 'ED Swap';
leg_str{5} = 'BB';

figure(1); 
subplot(2,2,1)
cla; hold all; grid on;
plot(N_vec,100*(1-mean(DropPercent.EST,1)),'-v')
plot(N_vec,100*(1-mean(DropPercent.EstSwap,1)),'-s')
plot(N_vec,100*(1-mean(DropPercent.ED,1)),'-d')
plot(N_vec,100*(1-mean(DropPercent.EdSwap,1)),'-o')
plot(N_vec,100*(1-mean(DropPercent.BB,1)),'-o')


xlabel('Number of Tasks')
ylabel('Percent of tasks scheduled')
title(sprintf('Monte = %i',MONTE))
legend(leg_str)


% figure(2); 
subplot(2,2,2)
cla; hold all; grid on
plot(N_vec,mean(Cost.EST,1),'-v')
plot(N_vec,mean(Cost.EstSwap,1),'-s')
plot(N_vec,mean(Cost.ED,1),'-d')
plot(N_vec,mean(Cost.EdSwap,1),'-o')
plot(N_vec,mean(Cost.BB,1),'-o')


xlabel('Number of Tasks')
ylabel('Average Cost (K = 4 Channels)')
legend(leg_str)


% figure(3); 
subplot(2,2,[3:4])
cla; grid on; hold all;
plot(N_vec,mean(RunTime.EST,1),'-v')
plot(N_vec,mean(RunTime.EstSwap,1),'-s')
plot(N_vec,mean(RunTime.ED,1),'-d')
plot(N_vec,mean(RunTime.EdSwap,1),'-o')
plot(N_vec,mean(RunTime.BB,1),'-o')
xlabel('Number of Tasks')
ylabel(sprintf('RunTime (seconds) (K = %i Channels)', K )) 
legend(leg_str)

%%

if FLAG.profile
    profile viewer
end
