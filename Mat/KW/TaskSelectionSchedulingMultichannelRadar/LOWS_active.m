function LowsFlag = LOWS_active(K,Tprime,PFprime,DRprime,TaskChannel,s_task,w_task,deadline_task,drop_task,length_task,ChannelAvailableTimeInput)

% Determines whether a better scheduled can be created by swapping tasks 


[t_ex,x,ChannelAvailableTimeProposed,TaskChannel] = BBMultiChannelSequenceScheduler(Tprime,s_task,deadline_task,length_task,ChannelAvailableTimeInput);
Cprime = sum( w_task(Tprime).*(t_ex(Tprime) - s_task(Tprime)) )  + sum( drop_task(DRprime) ) ;

CompletionTime = t_ex(Tprime) + length_task(Tprime);





for curChannel = 1:K
    ScheduledChannels = TaskChannel(Tprime);
    AdjIndex =   find( ScheduledChannels == curChannel,1,'last');
    if ~isempty(AdjIndex)
        AdjChannel(curChannel,1) = Tprime(AdjIndex);
    end
end

LowsFlag = 1;
for jj = 1:length(PFprime)
    
    curTask = PFprime(jj);
    Tnew = [Tprime(1:end-1); curTask];
    
    [t_new,x,ChannelAvailableTimeProposed,TaskChannel] = BBMultiChannelSequenceScheduler(Tnew,s_task,deadline_task,length_task,ChannelAvailableTimeInput);
    CompletionTimeNew = t_new(Tnew) + length_task(Tnew);
    Cnew = sum( w_task(Tnew).*(t_new(Tnew) - s_task(Tnew)) )  + sum( drop_task(DRprime) ) ;


    if CompletionTimeNew(end) < CompletionTime(end) % Better Completion Time
        if Cnew < Cprime
            LowsFlag = 0;
            break;
        end
    end
    
end



% for jj = 1:length(AdjChannel)
%     
%     AdjTask = AdjChannel(jj);
%     
%     %         Tnew = [
%     
%     
%     
%     
%     for ii = 1:length(PFprime)
%         task = PFprime(ii);
%         
%         
%         
%         
%     end
%     
% end




% Tproposed = [Tprime(1) sigma];
% [t_proposed] = BBActiveSequenceScheduler(Tproposed,s_task,deadline_task,length_task,ChannelAvailableTimeInput,ChannelSelected);
% Cprime = sum( w_task(Tproposed).*(t_proposed(Tproposed) - s_task(Tproposed)) )  + sum( drop_task(DRprime) ) ;