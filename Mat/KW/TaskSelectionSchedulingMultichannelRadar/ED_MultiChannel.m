function [T,C,t_ex,NumDropTask] = ED_MultiChannel(N,K,s_task,w_task,deadline_task,length_task,drop_task)


ChannelAvailableTime = zeros(K,1); % List of the times that the channel is available
[~,T] = sort(deadline_task); % Sort jobs based on starting times
t_ex = zeros(N,1);
x = zeros(N,1);
for ii = 1:N
    curJobId = T(ii);
    
    % Find Earliest Available Channel Time
    [AvailableTime,SelectedChannel] = min(ChannelAvailableTime);
    
    % Proposed: Place job on selected channel at approriate time
    t_ex(curJobId) = max( AvailableTime, s_task(curJobId) );
    
    % See if proposal results in dropped task
    x(curJobId) = (t_ex(curJobId) < deadline_task(curJobId));
    if x(curJobId) == 1 % Job is Scheduled update timeline, otherwise timeline can be used for another task      
        % Update Channels time availability
        ChannelAvailableTime(SelectedChannel) = t_ex(curJobId) + length_task(curJobId);        
%     else
%         keyboard
    end
    
    
end

x = (t_ex < deadline_task); % x represent whether tasks have been scheduled (1) or dropped (0)
C = sum( x.*w_task.*(t_ex - s_task) + (1 - x).*drop_task);

NumDropTask = N - sum(x);