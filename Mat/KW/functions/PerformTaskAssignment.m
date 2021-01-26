function [loss,t_run,T,t_ex,ChannelAvailableTime] = PerformTaskAssignment(approach_string,IterAlg,data)


K = data.K;
ChannelAvailableTime = data.ChannelAvailableTime;
timeSec = data.timeSec;

switch approach_string{IterAlg}
    case 'EST'
        t_ES = tic;
        [loss,t_ex,NumDropTask,T,ChannelAvailableTime] = ESTalgorithm(data);
        t_run = toc(t_ES);
    case 'ED'
        tstart = tic;
        [loss,t_ex,NumDropTask,T,ChannelAvailableTime] = EdAlgorithm(data);
        t_run = toc(tstart);
    case 'NN'
        tstart = tic;
        [loss,t_ex,NumDropTask,T,ChannelAvailableTime] = NeuralNetSchedulerAlgorithm(data);
        t_run = toc(tstart);        
    case 'BB'
       
        if K == 1 && abs( ChannelAvailableTime(1)  - timeSec ) > 1e-4
            %                         keyboard
        end
        
        t_BB = tic;
        [loss,t_ex,NumDropTask,T,ChannelAvailableTime] = BbQueueAlgorithm(data);
        
%         [T,~,~] = BBschedulerQueueVersion(K,s_task,deadline_task,d_task,drop_task,w_task,ChannelAvailableTime);
        %                     [t_ex,loss2] = fcn_BB_NN_linear_FAST(s_task,d_task,w_task,mode_stack,timeSec);
        %                     [~,T2] = sort(t_ex);
%         [loss,t_ex,ChannelAvailableTime] = FlexDARMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,d_task,drop_task,ChannelAvailableTime,RP);
        %                     [loss,t_ex,ChannelAvailableTime] = FunctionMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,d_task,drop_task,ChannelAvailableTime);
        t_run = toc(t_BB);
        %                     [t_ex,loss,t_run,Xnow,Ynow] = fcn_BB_NN_linear(s_task,d_task,w_task,mode_stack,timeSec);
        %                 [t_ex,loss,t_run,Xnow,Ynow] = fcn_search{i_a}(s_task,d_task,l_task,timeSec);
    case 'NN_Multiple'
        t_NN = tic;
        T = fcn_InferenceMultipleTimelines_BB_NN(s_task,deadline_task,d_task,drop_task,w_task,N,net);
        %                     [loss,t_ex,ChannelAvailableTime] = FunctionMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,d_task,drop_task,ChannelAvailableTime);
        [loss,t_ex,ChannelAvailableTime] = FlexDARMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,d_task,drop_task,ChannelAvailableTime,RP);
        t_run = toc(t_NN);
        %                     [loss,t_ex,t_run] =  fcn_Inference_BB_NN_linear(s_task,d_task,w_task,N,net,timeSec);
    case 'NN_Single'
        
        
        if FLAG.check
            ChannelAvailableTimeInput = ChannelAvailableTime;
            [T,~,~] = BBschedulerQueueVersion(K,s_task,deadline_task,d_task,drop_task,w_task,ChannelAvailableTimeInput);
            %                     [t_ex,loss2] = fcn_BB_NN_linear_FAST(s_task,d_task,w_task,mode_stack,timeSec);
            %                     [~,T2] = sort(t_ex);
            [loss_BB(iter),t_ex,~] = FlexDARMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,d_task,drop_task,ChannelAvailableTimeInput,RP);
        end
        
        if N < 8
            s_task = [s_task; zeros(8-N,1)];
            d_task = [d_task; 1*ones(8-N,1)];
            w_task = [w_task; zeros(8-N,1)];
            Ninput = 8;
        end
        
        t_NN = tic;
        [~,t_ex,~] =  fcn_Inference_BB_NN_linear(s_task,d_task,w_task,Ninput,net,timeSec);
        [~,T] = sort(t_ex);
        
        [loss,t_ex,ChannelAvailableTime] = FlexDARMultiChannelSequenceScheduler(T(1:N),N,K,s_task(1:N),w_task(1:N),deadline_task(1:N),d_task(1:N),drop_task(1:N),ChannelAvailableTime,RP);
        %                     [loss,t_ex,ChannelAvailableTime] = FunctionMultiChannelSequenceScheduler(T,N,K,s_task,w_task,deadline_task,d_task,drop_task,ChannelAvailableTime);
        t_run = toc(t_NN);
        
        if FLAG.check
            if loss_BB(iter) > loss
                keyboard
            end
        end        
        
        %         [t_ex,loss,t_run] = fcn_ES_linear(s_task,d_task,w_task,timeSec);
        %         [t_ex,loss,t_run,Xnow,Ynow] = fcn_BB_NN_linear(s_task,d_task,w_task,mode_stack,timeSec);
        %         [loss,t_ex,t_run] =  fcn_Inference_BB_NN_linear(s_task,d_task,w_task,N,net,timeSec);
        
end