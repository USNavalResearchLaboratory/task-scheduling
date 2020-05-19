function [t_ex,x,ChannelAvailableTime,TaskChannel] = BBPartialSequenceScheduler(t_ex,x,T,s_task,deadline_task,length_task,ChannelAvailableTime,TaskChannel,ForcedChannel)

% This function takes a sequence given by T = [1, 2, 3, 6, 3, ... ] and
% associated job starting times, tardiness weights, deadlines, durations,
% and dropping casts and places this sequence onto K parallel machines 
%
% Outputs: 
% C - cost associated with sequence as executed
% t_ex - time of each jobs execution
% Number of dropped tasks


L = length(T);
N = length(s_task);
K = length(ChannelAvailableTime);
% ChannelAvailableTime = zeros(K,1); % List of the times that the channel is available
% t_ex = zeros(N,1);
% x = zeros(N,1);
% TaskChannel = zeros(N,1);
for ii = L-1:L
    curJobId = T(ii);
    
    if ii > L - 2
        AvailableTime = ChannelAvailableTime(ForcedChannel);
        SelectedChannel = ForcedChannel;
    else
        % Find Earliest Available Channel Time
        [AvailableTime,SelectedChannel] = min(ChannelAvailableTime);
    end
    
    % Proposed: Place job on selected channel at approriate time
    t_ex(curJobId) = max( AvailableTime, s_task(curJobId) );
    
    % See if proposal results in dropped task
    x(curJobId) = (t_ex(curJobId) < deadline_task(curJobId));
    if x(curJobId) == 1 % Job is Scheduled update timeline, otherwise timeline can be used for another task      
        % Update Channels time availability
        ChannelAvailableTime(SelectedChannel) = t_ex(curJobId) + length_task(curJobId);    
        TaskChannel( curJobId ) = SelectedChannel;
    else
        TaskChannel( curJobId ) = SelectedChannel;  % Task is dropped, but still update selectedChannel for active schedule checker KW 3/30/20
        ChannelAvailableTime(SelectedChannel) = t_ex(curJobId) + length_task(curJobId);    

%         keyboard
    end   
end

% x = (t_ex < deadline_task); % x represent whether tasks have been scheduled (1) or dropped (0)
% C = sum( x.*w_task.*(t_ex - s_task) + (1 - x).*drop_task);
% NumDropTask = N - sum(x);