import copy
import numpy as np
from util.generic import check_rng
# from tree_search import TreeNode
# from tree_search import TreeNodeBound


class BranchBoundNode:
    """ Node object for branch and bound

    Attributes
    ---------



    """
    # TODO: docstring describes properties as attributes. OK? Subclasses, too.

    # _tasks = []  # TODO: needs to be overwritten by invoking scripts... OK?
    # _ch_avail_init = []
    # _rng = None

    def __init__(self, N, K, ch_avail_init: float):
        # if self.n_tasks == 0 or self.n_ch == 0:
        #     raise AttributeError("Cannot instantiate objects before assigning "
        #                          "the '_tasks' and 'n_ch class attributes.")

        self.T = np.empty([0],dtype = int)
        self.PF = np.array(range(N))
        self.NS = np.empty([0],dtype = int)
        self.t_ex = np.zeros(N)
        self.ch_avail = copy.deepcopy(ch_avail_init)
        self.ch_avail_init = ch_avail_init
        self.BestCost = float('inf')
        self.BestSeq = np.empty([0],dtype = int)
        self.CompleteSolutionFlag = 0
        self.ChannelAssignment = np.zeros(N)

        # self._seq = []
        #
        # self._seq_rem = set(range(self.n_tasks))
        #
        # self._t_ex = np.full(self.n_tasks, np.nan)  # task execution times (NaN for unscheduled)
        # self._ch_ex = np.full(self.n_tasks, np.nan, dtype=np.int)  # task execution channels
        #
        # self._ch_avail = copy.deepcopy(self._ch_avail_init)  # timeline availability
        #
        # self._l_ex = 0.  # partial sequence loss
        #
        # self.seq = seq
        # self.PF = set(range(self.n_tasks))
        # self.NS = []
    def candidate_task(self, tasks: list):
        curTaskIndex = np.argmin([tasks[n].t_release for n in self.PF]).tolist()
        curTask = self.PF[curTaskIndex]
        return curTask, curTaskIndex

    def seq2schedule_one_step(self,tasks: list): # TODO: Finish this function
        L = len(self.T)
        curJobId = np.int(self.T[L-1]) # Pull the last job off the task sequence list

        # Find Earliest Available Channel Time
        AvailableTime = self.ch_avail.min()
        SelectedChannel = self.ch_avail.argmin()

        # Place job on selected channel at appropriate time
        self.t_ex[curJobId] = max(AvailableTime, tasks[curJobId].t_release)

        # Update Node channel availability and assignment
        self.ch_avail[SelectedChannel] = self.t_ex[curJobId] + tasks[curJobId].duration
        self.ChannelAssignment[curJobId] = SelectedChannel

    def seq2schedule_multi_step(self, tasks: list):  # TODO: Finish this function
        L = len(self.T)
        for n in range(len(self.T)):
            curJobId = np.int(self.T[n])  # Pull the last job off the task sequence list

            # Find Earliest Available Channel Time
            AvailableTime = self.ch_avail.min()
            SelectedChannel = self.ch_avail.argmin()

            # Place job on selected channel at appropriate time
            self.t_ex[curJobId] = max(AvailableTime, tasks[curJobId].t_release)

            # Update Node channel availability and assignment
            self.ch_avail[SelectedChannel] = self.t_ex[curJobId] + tasks[curJobId].duration
            self.ChannelAssignment[curJobId] = SelectedChannel

    def seq2schedule_active(self, tasks:list, ForcedChannel: int):
        L = len(self.T)

        for n in range( (L-2) , L):
            curJobId = np.int(self.T[n])  # Pull the last job off the task sequence list

            # Force task onto same channel
            AvailableTime = self.ch_avail[ForcedChannel]
            SelectedChannel = ForcedChannel

            # Place job on selected channel at appropriate time
            self.t_ex[curJobId] = max(AvailableTime, tasks[curJobId].t_release)

            # Update Node channel availability and assignment
            self.ch_avail[SelectedChannel] = self.t_ex[curJobId] + tasks[curJobId].duration
            self.ChannelAssignment[curJobId] = SelectedChannel

        #     if ii > L - 2
        #         AvailableTime = ChannelAvailableTime(ForcedChannel);
        #         SelectedChannel = ForcedChannel;
        #     else
        #
        #         [AvailableTime, SelectedChannel] = min(ChannelAvailableTime);
        #     end
        #
        #     % Proposed: Place
        #     job
        #     on
        #     selected
        #     channel
        #     at
        #     approriate
        #     time
        #     t_ex(curJobId) = max(AvailableTime, s_task(curJobId));
        #
        #     % See if proposal
        #     results in dropped
        #     task
        #     x(curJobId) = (t_ex(curJobId) < deadline_task(curJobId));
        #     if x(curJobId) == 1 % Job is Scheduled update timeline, otherwise timeline can be used for another task
        #     % Update
        #     Channels
        #     time
        #     availability
        #     ChannelAvailableTime(SelectedChannel) = t_ex(curJobId) + length_task(curJobId);
        #     TaskChannel(curJobId) = SelectedChannel;
        #
        # else
        # TaskChannel(curJobId) = SelectedChannel; % Task is dropped, but
        # still
        # update
        # selectedChannel
        # for active schedule checker KW 3 / 30 / 20
        # ChannelAvailableTime(SelectedChannel) = t_ex(curJobId) + length_task(curJobId);




    def check_schedule_active(self, tasks: list, TimeExecutionInput: float,  ChannelAvailableTimeInput: float, ChannelAssignmentInput: float ):
        # Initially assume tasks are active
        active_flag = 1  # TODO: Added active schedule checking. For now assume schedule is active
        # Check if schedule is active(PFprime checked to see if any can task be scheduled right before last
        # task in Tprime on same timeline without imposing a delay
        Tprime = self.T.copy()
        PartialSchedule = Tprime[0:-1] # Schedule Prior to last time step
        ChannelAssignment = self.ChannelAssignment.copy()
        ChannelSelected = ChannelAssignment[Tprime[-1]].astype(int)    # Last channel selected

        t_partial = TimeExecutionInput
        ChannelAvailableTimePartial = ChannelAvailableTimeInput
        PartialChannelAssignment = ChannelAssignmentInput
        # x_partial = ScheduledIndicatorInput
        N = len(tasks)
        K = len(self.ch_avail)
        ch_avail_init = self.ch_avail_init.copy()

        for jj in range(len(self.PF)):
            curTask = self.PF[jj]
            Tactive = np.append(PartialSchedule, [curTask,  Tprime[-1]])
            tempNode = BranchBoundNode(N,K,ch_avail_init)
            tempNode.t_ex = t_partial
            tempNode.T = Tactive
            tempNode.ch_avail = ChannelAvailableTimePartial
            tempNode.ChannelAssignment = PartialChannelAssignment

            tempNode.seq2schedule_active(tasks, ChannelSelected)
            t_active = tempNode.t_ex.copy()

        #
        #     [t_active] = BBPartialSequenceScheduler(t_partial, x_partial, Tactive, s_task, deadline_task, length_task,
        #                                         ChannelAvailableTimePartial, PartialChannelAssignment, ChannelSelected)
        #
        #
            if t_active[Tprime[-1]] <= self.t_ex[Tprime[-1]]: # (s_task(curTask) + length_task(curTask)) < t_ex(Tprime(end)):
                active_flag = 0
                break





        return active_flag

    def partial_sequence_cost(self, tasks:list, K:int):
        Cprime = 0
        N = len(tasks)
        # ch_avail = copy.deepcopy(self.ch_avail_init)
        ch_avail = self.ch_avail_init.copy()
        t_ex = np.zeros(N)
        ChannelAssignment = np.zeros(N)

        for n in range(len(self.T)):
            # print(n)
            curJobId = self.T[n]

            # Find Earliest Available Channel Time
            AvailableTime = ch_avail.min()
            SelectedChannel = ch_avail.argmin()

            # Place job on selected channel at appropriate time
            t_ex[curJobId] = max(AvailableTime, tasks[curJobId].t_release)

            # Update Node channel availability and assignment
            ch_avail[SelectedChannel] = t_ex[curJobId] + tasks[curJobId].duration
            ChannelAssignment[curJobId] = SelectedChannel

            Cprime += tasks[curJobId].loss_fcn(t_ex[curJobId])
        return Cprime

    # % Find Earliest Available Channel Time
    # [AvailableTime,SelectedChannel] = min(ChannelAvailableTime);
    #
    # % Proposed: Place job on selected channel at approriate time
    # t_ex(curJobId) = max( AvailableTime, s_task(curJobId) );




def branch_bound_rules(tasks: list, ch_avail_init: list, verbose=False, rng=None):
    """Branch and Bound algorithm.

    Parameters
    ----------
    tasks : list of TaskRRM
    ch_avail_init : list of float
        Channel availability times.
    verbose : bool
        Enables printing of algorithm state information.
    rng
        NumPy random number generator or seed. Default Generator if None.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    # Initialize Stack
    N = len(tasks)
    K = len(ch_avail_init)
    S = []
    S.append( BranchBoundNode(N,K,ch_avail_init) ) # Make stack a list
    UB = float("inf") # Upper Bound

    # Iterate
    while len(S)>0:
        curNode = S.pop()  # Extract Current Node
        S.append(curNode)  # Put curNode back on stack. Need to learn python better way to do this.

        # # Pull off data from end of stack
        # PF = S(end).PF
        # T = S(end).T
        # NS = S(end).NS
        # DR = S(end).DR
        # ND = S(end).ND
        # TimeExecutionInput = S(end).t_ex
        # ChannelAvailableTimeInput = S(end).ChannelAvailableTime
        # ChannelAssignmentInput = S(end).ChannelAssignment
        # ScheduledIndicatorInput = S(end).x # x indicates whether  channel has been scheduled(1) or not scheduled(0)

        if len(curNode.PF) > 0:
            # Choose Candidate Task to place on timeline
            curTask, curTaskIndex = curNode.candidate_task(tasks)
            curNode.PF = np.delete(curNode.PF, curTaskIndex) # Remove curTask from current node
            PFprime = np.append(curNode.PF,curNode.NS)
            NSprime = np.empty([0],dtype = int)
            Tprime = np.append(curNode.T, curTask)

            curNode.NS = np.append(curNode.NS, curTask) # Added curTask to Not Scheduled list
            newNode = copy.deepcopy(curNode) # Copy Current Node to a New Node
            # newNode.T = np.append(curNode.T,curTask)  # Append candidate task to sequence of new node
            newNode.T = Tprime  # Append candidate task to sequence of new node
            newNode.PF = PFprime
            newNode.NS = NSprime
            # task = PF(task_index);
            # PF(task_index) = [];
            # Tprime = [T; task];
            # PFprime = [PF; NS];
            # NSprime = [];
            # DRprime = DR;
            # NDprime = [];
            # NS = [NS; task];
            #
            # % Update S(end)
            # S(end).PF = PF;
            # S(end).NS = NS;


            # Evaluate Partial Schedule on Timeline
            newNode.seq2schedule_one_step(tasks)

            # Start Times Dominance Rule
            ExecutionTimeOffsetFlag = np.diff(np.append(0, newNode.t_ex[newNode.T]))
            if all(ExecutionTimeOffsetFlag>=0): # Execution times need to be increasing, otherwise there is a better schedule out there
                Cprime = newNode.partial_sequence_cost(tasks,K)
                # Cprime = 0
                # for n in range(len(newNode.T)):
                #     print(n)
                #     curJobId = newNode.T[n]
                #     Cprime += Cprime + tasks[curJobId](newNode.t_ex[curJobId])

                    # self._l_ex += self._tasks[n](self._t_ex[n])
                    # ch = self.ch_early
                    # self._ch_ex[n] = ch
                    # self._t_ex[n] = max(self._tasks[n].t_release, self._ch_avail[ch])
                    # self._ch_avail[ch] = self._t_ex[n] + self._tasks[n].duration
                    # self._l_ex += self._tasks[n](self._t_ex[n])

                TimeExecutionInput = curNode.t_ex.copy()
                ChannelAvailableTimeInput = curNode.ch_avail.copy()
                ChannelAssignmentInput = curNode.ChannelAssignment.copy()
                active_flag = newNode.check_schedule_active(tasks, TimeExecutionInput, ChannelAvailableTimeInput, ChannelAssignmentInput)
                # active_flag = 1

                if active_flag:


                    if len(newNode.PF) == 0:
                        a = 1 # TODO Fill in this logic
                    else: #
                        a = 1 # TODO Fill in this logic

                    if Cprime < UB: # Add newNode to stack and continue
                        S.append(newNode)


                # Cprime = newNode
                # ExecutionTimeOffsetFlag = diff([0;  t_ex(Tprime)]) >= 0;
                # if all(ExecutionTimeOffsetFlag) % Start Times domainance rule

        else: # TODO: Finish the else statement
            # TODO Append NS to [T; NS], then evaluate t_ex for new list
            Ttemp = np.append(curNode.T, curNode.NS) # Append any entries from not scheduled list to current task list
            curNode.T = Ttemp # Update current node
            # curNode.ch_avail = np.zeros(K) # Reset Channel Availability (In the future this needs to be an input)
            # curNode.seq2schedule_multi_step(tasks)

            C = curNode.partial_sequence_cost(tasks,K)

            if np.size(curNode.NS) == 0 and C < UB: # Update Best Solution found so far
                UB = C
                Tstar = curNode.T
                t_ex_star = curNode.t_ex
                ch_ex_star = curNode.ChannelAssignment

                C2 = 0
                for n in range(len(tasks)):
                    C2 += tasks[n].loss_fcn(t_ex_star[n])
                a = 1


            if curNode.CompleteSolutionFlag == 1:
                if C < curNode.Best and curNode.T == N:
                    curNode.BestCost = C
                    curNode.BestSeq = curNode.T

            S.pop()

            # if isempty(NS) & & C < UB
            #     UB = C;
            #     Tstar = T;
            #     Tdr = DR;
            #     Tfinal = [T; DR];
            #     end
            #     if S(end).CompleteSolutionFlag == 1
            #         % S(end).CompleteSolutionFlag = 1;
            #         if C < S(end).BestCost & & length(T) == N
            #             S(end).BestCost = C;
            #             S(end).BestSeq = T;
            #             keyboard
            #         end
            #         NodeStats(NodeCnt) = S(end);
            #         NodeCnt = NodeCnt + 1;
            #         % keyboard
            #     end
            #     S(end) = [];



            # # Initialize Stack
            # UB = float("inf")
            # S.T = []
            #
            #
            # TreeNodePF._tasks = tasks
            # TreeNodePF._ch_avail_init = ch_avail
            # TreeNodePF._rng = check_rng(rng)
            #
            # stack = [TreeNodeBoundPF([])]  # Initialize Stack
            #
            # # node_best = stack[0].roll_out(do_copy=True)  # roll-out initial solution
            # # l_best = node_best.l_ex

            # Iterate
            # while len(stack) > 0:
            #
            #     node = stack.pop()  # Extract Node
            #     seq = copy.deepcopy(node.seq) # Current sequence of tasks in node
            #     PF = np.array(list(copy.deepcopy(node.PF)))  # Possible first tasks
            #     NS = np.array(list(copy.deepcopy(node.NS)))  # Not Scheduled Tasks
            #
            #     if len(PF) > 0:
            #         task_index = np.argmin([task.t_release for task in tasks]).tolist()
            #
            #         curTask = PF[curTask]
            #         PF = np.delete(PF,curTask)
            #         Tprime = np.append(seq,curTask)
            #         PFprime = PF
            #
            #         # Update Stack
            #         # stack(-1).
            #
            #     # node = stack[-1]
            #
            #
            #     # if verbose:
            #     #     print(f'# Remaining Nodes = {len(stack)}, Loss < {l_best:.3f}', end='\r')
            #     #
            #     #
            #     # # Branch
            #     # for node_new in node.branch(do_permute=True):  # TODO: check cutting! inequality?
            #     #     # Bound
            #     #     if node_new.l_lo < l_best:  # New node is not dominated
            #     #         if node_new.l_up < l_best:
            #     #             node_best = node_new.roll_out(do_copy=True)  # roll-out a new best node
            #     #             l_best = node_best.l_ex
            #     #             stack = [s for s in stack if s.l_lo < l_best]  # Cut Dominated Nodes
            #     #
            #     #         stack.append(node_new)  # Add New Node to Stack, LIFO

    # t_ex, ch_ex = node_best.t_ex, node_best.ch_ex  # optimal

    # dummy = BranchBoundNode(N,K)
    # dummy.T = Tstar
    # dummy.partial_sequence_cost(tasks)

    t_ex = 0
    ch_ex = 0

    return t_ex_star, ch_ex_star



class TreeNode:
    """
    Node object for tree search algorithms.

    Parameters
    ----------
    seq : list of list
        List of task index sequences by channel

    Attributes
    ----------
    seq : list of int
        Partial task index sequence.
    t_ex : ndarray
        Task execution times. NaN for unscheduled.
    ch_ex : ndarray
        Task execution channels. NaN for unscheduled.
    ch_avail : ndarray
        Channel availability times.
    l_ex : float
        Total loss of scheduled tasks.
    seq_rem: set
        Unscheduled task indices.

    """
    # TODO: docstring describes properties as attributes. OK? Subclasses, too.

    _tasks = []  # TODO: needs to be overwritten by invoking scripts... OK?
    _ch_avail_init = []
    _rng = None

    def __init__(self, seq: list):
        if self._n_tasks == 0 or self._n_ch == 0:
            raise AttributeError("Cannot instantiate objects before assigning "
                                 "the '_tasks' and 'n_ch class attributes.")

        self._seq = []

        self._seq_rem = set(range(self._n_tasks))

        self._t_ex = np.full(self._n_tasks, np.nan)  # task execution times (NaN for unscheduled)
        self._ch_ex = np.full(self._n_tasks, np.nan, dtype=np.int)  # task execution channels

        self._ch_avail = copy.deepcopy(self._ch_avail_init)  # timeline availability

        self._l_ex = 0.  # partial sequence loss

        self.seq = seq

    def __repr__(self):
        return f"TreeNode(sequence: {self.seq}, partial loss:{self.l_ex:.3f})"

    @property
    def _n_tasks(self):
        return len(self._tasks)

    @property
    def _n_ch(self):
        return len(self._ch_avail_init)

    @property
    def seq(self):
        """
        Gets the node sequence. Setter calls 'seq_ext'.

        Returns
        -------
        list of int
            Task index sequence

        """
        return self._seq

    @seq.setter
    def seq(self, seq):
        if len(seq) != len(set(seq)):
            raise ValueError("Input 'seq' must have unique values.")

        self.update_node(seq)

    @property
    def t_ex(self):
        return self._t_ex

    @property
    def ch_ex(self):
        return self._ch_ex

    @property
    def ch_avail(self):
        return self._ch_avail

    @property
    def ch_early(self):
        return int(np.argmin(self.ch_avail))

    @property
    def l_ex(self):
        return self._l_ex

    @property
    def seq_rem(self):
        return self._seq_rem

    def update_node(self, seq: list):
        """
        Sets node sequence using sequence-to-schedule approach.

        Parameters
        ----------
        seq : list of int
            Sequence of indices referencing cls._tasks.

        """

        if seq[:len(self._seq)] != self._seq:  # new sequence is not an extension of current sequence
            self.__init__(seq)  # initialize from scratch

        seq_append = seq[len(self._seq):]
        self._seq = seq
        self._seq_rem -= set(seq_append)
        for n in seq_append:
            ch = self.ch_early

            self._ch_ex[n] = ch
            self._t_ex[n] = max(self._tasks[n].t_release, self._ch_avail[ch])
            self._ch_avail[ch] = self._t_ex[n] + self._tasks[n].duration
            self._l_ex += self._tasks[n].loss_fcn(self._t_ex[n])

    def branch(self, do_permute=True):
        """
        Generate descendant nodes.

        Parameters
        ----------
        do_permute : bool
            Enables random permutation of returned node list.

        Yields
        -------
        TreeNode
            Descendant node with one additional task scheduled.

        """

        seq_iter = list(self._seq_rem)
        seq_iter_array = np.array(seq_iter)
        if do_permute:
            # self._rng.shuffle(seq_iter)
            t_release = np.array([task.t_release for task in self._tasks])
            seq_iter = list(seq_iter_array[np.argsort(t_release[seq_iter])])
        # self.seq = list(np.argsort([task.t_release for task in self._tasks]))

        for n in seq_iter:
            # seq_new = copy.deepcopy(self.seq)
            seq_new = self.seq.copy()

            seq_new.append(n)

            node_new = copy.deepcopy(self)  # new TreeNode object
            node_new.seq = seq_new  # call seq.setter method

            yield node_new

    def roll_out(self, do_copy=False):
        """
        Generates/updates node with a randomly completed sequence.

        Parameters
        ----------
        do_copy : bool
            Enables return of a new TreeNode object. Otherwise, updates in-place.

        Returns
        -------
        TreeNode
            Only if do_copy is True.

        """

        # seq_new = copy.deepcopy(self.seq) + self._rng.permutation(list(self._seq_rem)).tolist()
        seq_new = self.seq.copy() + self._rng.permutation(list(self._seq_rem)).tolist()


        if do_copy:
            node_new = copy.deepcopy(self)  # new TreeNode object
            node_new.seq = seq_new  # call seq.setter method

            return node_new
        else:
            self.seq = seq_new  # call seq.setter method

    def check_swaps(self):
        """Try adjacent task swapping, overwrite node if loss drops."""

        if len(self.seq_rem) != 0:
            raise ValueError("Node sequence must be complete.")

        for i in range(len(self.seq) - 1):
            seq_swap = copy.deepcopy(self.seq)
            seq_swap[i:i + 2] = seq_swap[i:i + 2][::-1]
            node_swap = TreeNode(seq_swap)
            if node_swap.l_ex < self.l_ex:
                self = node_swap            # TODO: improper?


class TreeNodeBound(TreeNode):
    """
    Node object with additional loss bounding attributes.

        Parameters
        ----------
        seq : list of list
            List of task index sequences by channel

        Attributes
        ----------
        seq : list of int
            Partial task index sequence.
        t_ex : ndarray
            Task execution times. NaN for unscheduled.
        ch_ex : ndarray
            Task execution channels. NaN for unscheduled.
        ch_avail : ndarray
            Channel availability times.
        l_ex : float
            Total loss of scheduled tasks.
        seq_rem: set
            Unscheduled task indices.
        l_lo: float
            Lower bound on total loss for descendant nodes.
        l_up: float
            Upper bound on total loss for descendant nodes.

        """

    def __init__(self, seq: list):
        self._l_lo = 0.
        self._l_up = float('inf')
        super().__init__(seq)

    def __repr__(self):
        return f"TreeNodeBound(sequence: {self.seq}, {self.l_lo:.3f} < loss < {self.l_up:.3f})"

    @property
    def l_lo(self):
        return self._l_lo

    @property
    def l_up(self):
        return self._l_up

    def update_node(self, seq: list):
        """
        Sets node sequence and iteratively updates all dependent attributes.

        Parameters
        ----------
        seq : list of list
            Sequence of indices referencing cls._tasks.

        """

        super().update_node(seq)

        # Add bound attributes
        t_ex_max = (max([self._tasks[n].t_release for n in self._seq_rem] + list(self._ch_avail))
                    + sum([self._tasks[n].duration for n in self._seq_rem]))  # maximum execution time for bounding

        self._l_lo = self._l_ex
        self._l_up = self._l_ex
        for n in self._seq_rem:  # update loss bounds
            self._l_lo += self._tasks[n].loss_fcn(max(self._tasks[n].t_release, min(self._ch_avail)))
            self._l_up += self._tasks[n].loss_fcn(t_ex_max)

        if len(self._seq_rem) > 0 and self._l_lo == self._l_up:  # roll-out if bounds converge
            self.roll_out()


def branch_bound2(tasks: list, ch_avail: list, verbose=False, rng=None):
    """
    Branch and Bound algorithm.

    Parameters
    ----------
    tasks : list of GenericTask
    ch_avail : list of float
        Channel availability times.
    verbose : bool
        Enables printing of algorithm state information.
    rng
        NumPy random number generator or seed. Default Generator if None.

    Returns
    -------
    t_ex : ndarray
        Task execution times.
    ch_ex : ndarray
        Task execution channels.

    """

    TreeNode._tasks_init = tasks
    TreeNode._ch_avail_init = ch_avail
    TreeNode._rng = check_rng(rng)

    stack = [TreeNodeBound([])]  # Initialize Stack

    node_best = stack[0].roll_out(do_copy=True)  # roll-out initial solution
    l_best = node_best.l_ex

    # Iterate
    while len(stack) > 0:
        node = stack.pop()  # Extract Node -- Pop first element. Added est to branch function therefore first elements should be most promising

        # Branch
        for node_new in node.branch(do_permute=True):
            # Bound
            if node_new.l_lo < l_best:  # New node is not dominated
                ExecutionTimeOffsetFlag = np.diff(np.append(0, node_new.t_ex[node_new.seq]))
                if all(ExecutionTimeOffsetFlag >= 0):  # Execution times need to be increasing, otherwise there is a better schedule out there: KW
                    # Add Active Schedule Check
                    if node_new.l_up < l_best:
                        node_best = node_new.roll_out(do_copy=True)  # roll-out a new best node
                        l_best = node_best.l_ex
                        stack = [s for s in stack if s.l_lo < l_best]  # Cut Dominated Nodes

                    stack.append(node_new)  # Add New Node to Stack, LIFO

        if verbose:
            progress = 1 - sum([math.factorial(len(node.seq_rem)) for node in stack]) / math.factorial(len(tasks))
            print(f'Search progress: {100*progress:.1f}% - Loss < {l_best:.3f}', end='\r')
            # print(f'# Remaining Nodes = {len(stack)}, Loss < {l_best:.3f}', end='\r')

    t_ex, ch_ex = node_best.t_ex, node_best.ch_ex  # optimal

    return t_ex, ch_ex




def stats2nnXY(NodeStats, tasks: list):

    # Prune Any redundent Schedules
    NodeStats = prune_NodeStats(NodeStats)

    # Feature Data
    t_release = np.array([task.t_release for task in tasks])
    t_drop = np.array([task.t_drop for task in tasks])
    duration = np.array([task.duration for task in tasks])
    cost_drop = np.array([task.l_drop for task in tasks])
    slope = np.array([task.slope for task in tasks])
    refTime = 0

    # Generate Policy Features
    N = len(tasks)
    PF = np.empty([5+N+3,N])
    PF[0,:] = t_release - refTime # Make first feature relative to minimum start time
    PF[1,:] = t_drop - refTime # Make 2nd feature relative to minimum start time
    PF[2,:] = duration
    PF[3,:] = cost_drop
    PF[4,:] = slope

    SeqCost = np.array([node.l_ex for node in NodeStats])

    sortIdx = np.argsort(SeqCost) # Organize Costs in Ascending order
    OptimalSeq = NodeStats[sortIdx[0]].seq

    # Expand Children of All NodeStats
    cnt = 1
    RecordSeq = np.zeros([N, 1])
    CostFinal = np.array([0])
    zeroVec = np.zeros([N,1])
    Y = []
    X = []

    for jj in sortIdx:
        curNode = NodeStats[jj]
        BestSeq = curNode.seq
        for kk in range(N):
            node = np.array(BestSeq[0:kk])
            optimal_action = BestSeq[kk]
            T = node.copy()
            new_seq = np.transpose( [np.append(T, -1*np.ones([N - len(T), 1]) )] )

            if cnt == 1:
                VisitIndicator = 0
            else:
                SeqDiff = (RecordSeq - new_seq)  # TODO: Start Here
                # SeqDiff = bsxfun( @ plus, RecordSeq, -new_seq );
                # VisitIndicator = sum(sum(SeqDiff == zeroVec, 1) == N);
                # VisitIndicator = sum(~any(SeqDiff))
                # temp = np.reshape(SeqDiff, -1)
                # VisitIndicator = not(any(temp))
                VisitIndicator = not(all(SeqDiff.any(axis=0)))
                # % if VisitIndicator ~= VisitIndicator2
                # % keyboard
                # % end

            if VisitIndicator == 0:
                if cnt == 1:
                    RecordSeq[:, cnt-1] = new_seq[:,0]
                    CostFinal[cnt - 1] = curNode.l_ex
                else:
                    RecordSeq = np.append(RecordSeq, new_seq,1)
                    CostFinal = np.append(CostFinal, curNode.l_ex)

                Y.append(optimal_action)
                # Y[cnt-1] = optimal_action.copy()

                PFtree = np.zeros([N, N])

                for ind1 in range(len(node)):
                    ind2 = node[ind1]
                    PFtree[ind1,ind2] = 1

                # IND = sub2ind([N N], [1:length(node)]',node);
                # PFtree[IND] = 1

                PfStatus = np.zeros([3, N])
                PfStatus[0,:] = 1
                for nn in range(len(node)):
                    PfStatus[:, node[nn] ] = [0, 0, 1] # Infeasible Already Assigned

                PF[5:5+N,:] = PFtree
                PF[5+N:N+8,:] = PfStatus
                A = copy.deepcopy(PF)
                X.append(A)
                # X[:,:, cnt] = [PF, PFtree, PfStatus]

                # Timeline ignored for now -
                # KTW 13APR2020 - flipped PFtree and PFStatus location.Need to
                # match NeuralNetSchedulerAlgorithm.As long as they both match
                # everything will work better.My personal preference is
                # location you are in tree, then what nodes you visited, but
                # again doesn't matter as long as it's consistent with
                # implementation of NN - scheduler

                # Note: Could find if node is dominated by calculating cost at the partial sequence and comparing
                # across other partial nodes. Not doing at the moment.
                cnt = cnt + 1

    return X,Y


def prune_NodeStats(NodeStats):

    A = np.array([node.seq  for node in NodeStats])
    N = len(A)
    for jj in range(N):
        row = A[jj,:]
        equality = np.all(A == row, axis=1)
        equality[jj] = False # Same row will always be true
        if any(equality):
            a = 1 # TODO: Doesn't do anything right now, not a problem as stats2NNxy will remove redundacy, but this could speed this up.





    return NodeStats