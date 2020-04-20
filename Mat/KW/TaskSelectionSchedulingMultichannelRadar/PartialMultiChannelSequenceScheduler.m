function [C,t_ex,NumDropTask] = PartialMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,length_task,drop_task)

% Code is deprecated. Why?? Because it always sets ChannelAvailableTime to
% zero. Not good. Use "FunctionMultiChannelSequenceScheduler" instead.

% This function takes a sequence given by T = [1, 2, 3, 6, 3, ... ] and
% associated job starting times, tardiness weights, deadlines, durations,
% and dropping casts and places this sequence onto K parallel machines 
%
% Outputs: 
% C - cost associated with sequence as executed
% t_ex - time of each jobs execution
% Number of dropped tasks


ChannelAvailableTime = zeros(K,1); % List of the times that the channel is available
t_ex = zeros(N,1);
x = zeros(N,1);
% repeat_check = 1;
for ii = 1:length(T)
    curJobId = T(ii);
    
    % Find Earliest Available Channel Time
%     if repeat_check == 1

        [AvailableTime,SelectedChannel] = min(ChannelAvailableTime);
%         MinIndex = find( AvailableTime == ChannelAvailableTime );
%     else
        
%     end
    
    
    % Proposed: Place job on selected channel at approriate time
    t_ex(curJobId) = max( AvailableTime, s_task(curJobId) );
    
    % See if proposal results in dropped task
    x(curJobId) = (t_ex(curJobId) < deadline_task(curJobId));
    if x(curJobId) == 1 % Job is Scheduled update timeline, otherwise timeline can be used for another task      
        % Update Channels time availability
        ChannelAvailableTime(SelectedChannel) = t_ex(curJobId) + length_task(curJobId);        
    else
        ChannelAvailableTime(SelectedChannel) = t_ex(curJobId) + length_task(curJobId);
%         keyboard
    end   
end

% x = (t_ex < deadline_task); % x represent whether tasks have been scheduled (1) or dropped (0)
C = sum( x(T).*w_task(T).*(t_ex(T) - s_task(T)) + (1 - x(T)).*drop_task(T));
NumDropTask = length(T) - sum(x(T));