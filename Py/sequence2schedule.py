
import numpy as np

def FlexDARMultiChannelSequenceScheduler(T: list, tasks: list, ch_avail: list, RP):

    """FlexDARMultiChannelSequenceScheduler

        #T ,N ,K ,s_task ,w_task,deadline_task ,length_task,drop_task,ChannelAvailableTime ,RP)


       Parameters
       ----------
       T :  Sequence of tasks to place onto timeline
       tasks : list of Generic
       ch_avail : list of float
           Channel availability times.
       RP : Resource Period to schedule onto
       ChannelAvailableTime : Floating point number that indicates the incoming channel available time on both timelines

       exhaustive : bool
           Enables an exhaustive tree search. If False, sequence-to-schedule assignment is used.
       verbose : bool
           Enables printing of algorithm state information.
       rng
           NumPy random number generator.

       Returns
       -------
       t_ex : ndarray
           Task execution times.
       ch_ex : ndarray
           Task execution channels.

    function [C ,t_ex ,ChannelAvailableTime ,NumDropTask] = FlexDARMultiChannelSequenceScheduler(T ,N ,K ,s_task ,w_task
                                                                                                ,deadline_task ,length_task
                                                                                                ,drop_task
                                                                                                ,ChannelAvailableTime ,RP)

    % This function takes a sequence given by T = [1, 2, 3, 6, 3, ... ] and
    % associated job starting times, tardiness weights, deadlines, durations,
    % and dropping casts and places this sequence onto K parallel machines
    %
    % Inputs: ChannelAvailableTime is an input
    %
    % Outputs:
    % C - cost associated with sequence as executed
        % t_ex - time of each jobs execution
    % Number of dropped tasks


    % ChannelAvailableTime = zeros(K ,1); % List of the times that the channel is available
    t_ex = zeros(N ,1);
    x = zeros(N ,1);
    % repeat_check = 1;
    for ii = 1:length(T)
    curJobId = T(ii);

    % Find Earliest Available Channel Time
    %     if repeat_check == 1

        [AvailableTime ,SelectedChannel] = min(ChannelAvailableTime);
    %         MinIndex = find( AvailableTime == ChannelAvailableTime );
    %     else

    %     end


    % Proposed: Place job on selected channel at approriate time
    TaskStartTime = max( AvailableTime, s_task(curJobId) );
    TaskFinishTime = TaskStartTime + length_task(curJobId);
    % Check to make sure Task is completed in same resource period. If not
    % move TaskStartTime to next resource period. This assumes no tasks
    % are longer than one RP. It will break if this is the case
    if floor(TaskStartTim e /RP) == floor(TaskFinishTim e /RP)
        t_ex(curJobId) = TaskStartTime;
    else
        t_ex(curJobId) = floor(TaskFinishTim e /RP ) *RP;
    end

    %     t_ex(curJobId) = max( AvailableTime, s_task(curJobId) );

    % See if proposal results in dropped task
    x(curJobId) = (t_ex(curJobId) < deadline_task(curJobId));
    if x(curJobId) == 1 % Job is Scheduled update timeline, otherwise timeline can be used for another task
    % Update Channels time availability
    ChannelAvailableTime(SelectedChannel) = t_ex(curJobId) + length_task(curJobId);
    else % Still place on timeline. Delay penalty is captured in cost function line 62. KW 4/ 02 / 2020
    ChannelAvailableTime(SelectedChannel) = t_ex(curJobId) + length_task(curJobId);
    % keyboard
    end
    % if x(curJobId) == 0
        % keyboard
    % end

    end

    % x = (t_ex < deadline_task); % x
    represent
    whether
    tasks
    have
    been
    scheduled(1) or dropped(0)
    C = sum(x. * w_task. * (t_ex - s_task) + (1 - x). * drop_task);
    NumDropTask = N - sum(x);

    """

    N = len(tasks)
    t_ex = np.zeros(N)
    ch_ex = np.zeros(N) #      channel assigment. Index indicates which channel task was assigned too ch_ex = [0 1 1] implies task 1 assigned to channel 0, etc...
    x = np.zeros(N) # Tracks whether tasks were assigned or dropped
    t_release = np.zeros(N)
    duration = np.zeros(N)
    deadline_task = np.zeros(N)


    for n in range(N):
        t_release[n] = tasks[n].t_release
        duration[n] = tasks[n].duration


    for ii in range(N):
        curJobId = T[ii]
        AvailableTime = np.min(ch_avail)
        SelectedChannel = np.argmin(ch_avail)
        ch_ex[curJobId] = SelectedChannel
        # Proposed: Place job on selected channel at approriate time
        TaskStartTime = max(AvailableTime, t_release[curJobId])
        TaskFinishTime = TaskStartTime + duration[curJobId]

        # Check to make sure Task is completed in same resource period. If not
        # move TaskStartTime to next resource period. This assumes no tasks
        # are longer than one RP. It will break if this is the case
        if np.floor(TaskStartTime / RP) == np.floor(TaskFinishTime / RP):
            t_ex[curJobId] = TaskStartTime
        else:
            t_ex[curJobId] = np.floor(TaskFinishTime / RP ) *RP

        # See if proposal results in dropped task
        x[curJobId] = (t_ex[curJobId] < deadline_task[curJobId])
        if x[curJobId] == 1:  # Job is Scheduled update timeline, otherwise timeline can be used for another task
        # Update Channels time availability
            ch_avail[SelectedChannel] = t_ex[curJobId] + duration[curJobId]
        else:  # Still place on timeline. Delay penalty is captured in cost function line 62. KW 4/ 02 / 2020
            ch_avail[SelectedChannel] = t_ex[curJobId] + duration[curJobId]
        # keyboard


    # FROM MATLAB

    # # # repeat_check = 1;
    # #for ii = 1:length(T)
    # #curJobId = T(ii);
    #
    # # Find Earliest Available Channel Time
    # #     if repeat_check == 1
    #
    #     [AvailableTime ,SelectedChannel] = min(ChannelAvailableTime);
    # #         MinIndex = find( AvailableTime == ChannelAvailableTime );
    # #     else
    #
    # #     end
    #
    #
    # # Proposed: Place job on selected channel at approriate time
    # TaskStartTime = max( AvailableTime, s_task(curJobId) );
    # TaskFinishTime = TaskStartTime + length_task(curJobId);
    # # Check to make sure Task is completed in same resource period. If not
    # # move TaskStartTime to next resource period. This assumes no tasks
    # # are longer than one RP. It will break if this is the case
    # if floor(TaskStartTim e /RP) == floor(TaskFinishTim e /RP)
    #     t_ex(curJobId) = TaskStartTime;
    # else
    #     t_ex(curJobId) = floor(TaskFinishTim e /RP ) *RP;
    # end
    #
    # #     t_ex(curJobId) = max( AvailableTime, s_task(curJobId) );
    #
    # # See if proposal results in dropped task
    # x(curJobId) = (t_ex(curJobId) < deadline_task(curJobId));
    # if x(curJobId) == 1 # Job is Scheduled update timeline, otherwise timeline can be used for another task
    # # Update Channels time availability
    # ChannelAvailableTime(SelectedChannel) = t_ex(curJobId) + length_task(curJobId);
    # else # Still place on timeline. Delay penalty is captured in cost function line 62. KW 4/ 02 / 2020
    # ChannelAvailableTime(SelectedChannel) = t_ex(curJobId) + length_task(curJobId);
    # # keyboard
    # end
    # # if x(curJobId) == 0
    #     # keyboard
    # # end
    #
    # end
    #
    # # x = (t_ex < deadline_task); # x represent whether tasks have been scheduled(1) or dropped(0)
    # C = sum(x. * w_task. * (t_ex - s_task) + (1 - x). * drop_task);
    # NumDropTask = N - sum(x);

    return t_ex, ch_ex