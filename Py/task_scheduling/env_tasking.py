import time
from types import MethodType
from functools import partial
import dill

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.spaces import Box, Space
# from baselines import deepq

from util.plot import plot_task_losses
from generators.scheduling_problems import Random as RandomProblem
from tree_search import TreeNode, TreeNodeShift


# Gym Spaces
class Sequence(Space):
    """Gym Space for index sequences."""

    def __init__(self, n):
        self.n = n      # sequence length
        super().__init__((n,), np.int)

    def sample(self):
        return self.np_random.permutation(self.n)

    def contains(self, x):
        return True if (np.sort(np.asarray(x, dtype=int)) == np.arange(self.n)).all() else False

    def __repr__(self):
        return f"Sequence({self.n})"

    def __eq__(self, other):
        return isinstance(other, Sequence) and self.n == other.n


class DiscreteSet(Space):
    """Gym Space for discrete, non-integral elements."""

    def __init__(self, elements):
        self.elements = np.sort(np.asarray(list(elements)).flatten())   # ndarray representation of set
        self.n = self.elements.size
        super().__init__((self.n,), self.elements.dtype)

    def sample(self):
        return self.np_random.choice(self.elements)

    def contains(self, x):
        return True if x in self.elements else False

    def __repr__(self):
        return f"DiscreteSet({self.elements})"

    def __eq__(self, other):
        return isinstance(other, DiscreteSet) and self.elements == other.elements


# Gym Environments

class BaseTaskingEnv(gym.Env):
    """
    Base environment for task scheduling.

    Parameters
    ----------
    problem_gen : generators.scheduling_problems.Base
        Scheduling problem generation object.
    node_cls : TreeNode or callable
        Class for tree search node generation.
    features : ndarray, optional
        Structured numpy array of features with fields 'name', 'func', and 'lims'.
    sort_func : function or str, optional
        Method that returns a sorting value for re-indexing given a task index 'n'.

    """

    def __init__(self, problem_gen, node_cls=TreeNode, features=None, sort_func=None):
        self.problem_gen = problem_gen
        self.solution = None

        self.n_tasks = self.problem_gen.n_tasks
        self.n_ch = self.problem_gen.n_ch

        # Set features and state bounds
        if features is not None:
            self.features = features
        else:
            _task_param_names = self.problem_gen.task_gen.cls_task.param_names
            self.features = np.array(list(zip(_task_param_names,
                                              [lambda task, name=_name: getattr(task, name)
                                               for _name in _task_param_names],     # note: late-binding closure
                                              self.problem_gen.task_gen.param_lims.values())),
                                     dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

            # self.features = np.array([], dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])
            # _low, _high = self.problem_gen.task_gen.param_repr_lim

        _low, _high = zip(*self.features['lims'])
        self._state_tasks_low = np.broadcast_to(_low, (self.n_tasks, len(_low)))
        self._state_tasks_high = np.broadcast_to(_high, (self.n_tasks, len(_high)))

        # Set sorting method
        if callable(sort_func):
            self.sort_func = MethodType(sort_func, self)
        elif type(sort_func) == str:
            def _sort_func(env, n):
                return getattr(env.tasks[n], sort_func)

            self.sort_func = MethodType(_sort_func, self)
        else:
            self.sort_func = None

        self.masking = False

        self.reward_range = (-float('inf'), 0)

        self.node_cls = node_cls
        self.node = None

    tasks = property(lambda self: self.node.tasks)
    ch_avail = property(lambda self: self.node.ch_avail)

    @property
    def sorted_index(self):
        """Indices for task re-ordering for environment state."""
        if callable(self.sort_func):
            return np.argsort([self.sort_func(n) for n in range(self.n_tasks)])
        else:
            return np.arange(self.n_tasks)

    @property
    def state_tasks(self):
        """State sub-array for task features."""
        state_tasks = np.array([task.gen_features(*self.features['func']) for task in self.tasks])
        if self.masking:
            state_tasks[self.node.seq] = 0.     # zero out state rows for scheduled tasks

        return state_tasks[self.sorted_index]       # sort individual task states

    def reset(self, tasks=None, ch_avail=None, persist=False, solve=False):
        """
        Reset environment by re-initializing node object with random (or user-specified) tasks/channels.

        Parameters
        ----------
        tasks : Sequence of tasks.Generic, optional
            Optional task set for non-random reset.
        ch_avail : Sequence of float, optional
            Optional initial channel availabilities for non-random reset.
        persist : bool
            If True, keeps tasks and channels fixed during reset, regardless of other inputs.
        solve : bool
            Solves and stores the Branch & Bound optimal schedule.

        """

        if not persist:     # use new random (or user-specified) tasks/channels

            if tasks is None or ch_avail is None:
                if solve:
                    ((tasks, ch_avail), self.solution), = self.problem_gen(solve=True)
                else:
                    (tasks, ch_avail), = self.problem_gen()
                    self.solution = None

            elif len(tasks) != self.n_tasks:
                raise ValueError(f"Input 'tasks' must be None or a list of {self.n_tasks} tasks")
            elif len(ch_avail) != self.n_ch:
                raise ValueError(f"Input 'ch_avail' must be None or an array of {self.n_ch} channel availabilities")

            self.node_cls._tasks_init = tasks
            self.node_cls._ch_avail_init = ch_avail

        self.node = self.node_cls()

    def step(self, action):
        """Updates environment state based on task index input."""
        action = self.sorted_index[action]  # decode task index to original order
        self.node.seq_extend(action)  # updates sequence, loss, task parameters, etc.

    def render(self, mode='human'):
        if mode == 'human':
            fig_env, ax_env = plt.subplots(num='Task Scheduling Env', clear=True)
            plot_task_losses(self.tasks, ax=ax_env)

    def close(self):
        plt.close('all')


class SeqTaskingEnv(BaseTaskingEnv):
    """Tasking environment with single action of a complete task index sequence."""

    @staticmethod
    def infer_action_space(observation):
        """Determines the action Gym.Space from an observation."""
        return Sequence(len(observation))

    def reset(self, tasks=None, ch_avail=None, persist=False, solve=False):
        super().reset(tasks, ch_avail, persist, solve)

        # Gym observation and action spaces
        self.observation_space = Box(self._state_tasks_low, self._state_tasks_high, dtype=np.float64)
        self.action_space = Sequence(self.n_tasks)

        return self.state_tasks     # observation is the task set state

    def step(self, action):
        """
        Updates environment state with a complete index sequence.

        Parameters
        ----------
        action : Sequence of int
            Complete index sequence.

        Returns
        -------
        observation : None
        reward : float
            Negative loss achieved by the complete sequence.
        done : True
            Episode completes after one step.
        info : dict
            Auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        """

        if len(action) != self.n_tasks:
            raise ValueError("Action must be a complete sequence.")

        super().step(action)
        reward = -1 * self.node.l_ex    # negative loss of complete sequence

        return None, reward, True, {}       # Episode is done, no observation


class StepTaskingEnv(BaseTaskingEnv):
    """
    Tasking environment with actions of single task indices.

    Parameters
    ----------
    problem_gen : generators.scheduling_problems.Base
        Scheduling problem generation object.
    node_cls : TreeNode or callable
        Class for tree search node generation.
    features : ndarray, optional
        Structured numpy array of features with fields 'name', 'func', and 'lims'.
    sort_func : function or str, optional
        Method that returns a sorting value for re-indexing given a task index 'n'.
    seq_encoding : function or str, optional
        Method that returns a 1-D encoded sequence representation for a given task index 'n'. Assumes that the
        encoded array sums to one for scheduled tasks and to zero for unscheduled tasks.
    masking : bool
        If True, features are zeroed out for scheduled tasks.

    """

    def __init__(self, problem_gen, node_cls=TreeNode, features=None, sort_func=None, seq_encoding='one-hot',
                 masking=False):

        # Set sequence encoder method
        if callable(seq_encoding):
            self.seq_encoding = MethodType(seq_encoding, self)
        elif type(seq_encoding) == str:     # simple string specification for supported encoders
            if seq_encoding == 'indicator':
                def _seq_encoding(env, n):
                    return [1] if n in env.node.seq else [0]
            elif seq_encoding == 'one-hot':
                def _seq_encoding(env, n):
                    out = np.zeros(env.n_tasks)
                    if n in env.node.seq:
                        out[env.node.seq.index(n)] = 1
                    return out
            else:
                raise ValueError("Unsupported sequence encoder string.")

            self.seq_encoding = MethodType(_seq_encoding, self)
        else:
            raise TypeError("Sequence encoding input must be callable or str.")

        self.loss_agg = None

        super().__init__(problem_gen, node_cls, features, sort_func)

        self.masking = masking

    @property
    def observation_space(self):
        """Gym space of valid observations."""
        _state_low = np.concatenate((np.zeros(self.state_seq.shape), self._state_tasks_low), axis=1)
        _state_high = np.concatenate((np.ones(self.state_seq.shape), self._state_tasks_high), axis=1)
        return Box(_state_low, _state_high, dtype=np.float64)

    @property
    def action_space(self):
        """Gym space of valid actions."""
        seq_rem_sort = np.flatnonzero(np.isin(self.sorted_index, list(self.node.seq_rem)))
        return DiscreteSet(seq_rem_sort)

    def infer_action_space(self, observation):
        """Determines the action Gym.Space from an observation."""
        _state_seq = observation[:, :-len(self.features)]
        return DiscreteSet(np.flatnonzero(1 - _state_seq.sum(1)))

    @property
    def state_seq(self):
        """State sub-array for encoded partial sequence."""
        state_seq = np.array([self.seq_encoding(n) for n in range(self.n_tasks)])
        return state_seq[self.sorted_index]  # sort individual sequence states

    @property
    def state(self):
        """Complete state."""
        return np.concatenate((self.state_seq, self.state_tasks), axis=1)

    def reset(self, tasks=None, ch_avail=None, persist=False, solve=False):
        super().reset(tasks, ch_avail, persist, solve)
        self.loss_agg = self.node.l_ex      # Loss can be non-zero due to time origin shift during node initialization

        return self.state

    def step(self, action: int):
        """
        Updates environment state with a single task index.

        Parameters
        ----------
        action : int
            Single task index.

        Returns
        -------
        observation : ndarray
            Complete state, including task parameters and sequence encoding.
        reward : float
            Negative incremental loss after scheduling a single task.
        done : bool
            True if node sequence is complete.
        info : dict
            Auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        """

        super().step(action)

        reward, self.loss_agg = self.loss_agg - self.node.l_ex, self.node.l_ex
        done = len(self.node.seq_rem) == 0      # sequence is complete

        return self.state, reward, done, {}


# Agents
class RandomAgent:
    """The world's simplest agent!"""
    def __init__(self, infer_action_space):
        self.infer_action_space = infer_action_space

    def act(self, observation, reward, done):
        action_space = self.infer_action_space(observation)
        return action_space.sample()       # randomly selected action


# Learning
def data_gen(env, n_gen=1, save=False, file=None):
    """
    Generate state-action data for learner training and evaluation.

    Parameters
    ----------
    env : gym.Env
        Gym environment
    n_gen : int
        Number of random tasking problems to generate data from.
    save : bool
        If True, data is saved in a tensorflow.data.Dataset object

    Returns
    -------
    x_set : ndarray
        Observable predictor data.
    y_set : ndarray
        Unobserved target data.

    """

    if not isinstance(env, StepTaskingEnv):
        raise NotImplementedError("Tasking environment must be step Env.")      # TODO: generalize for full seq

    x_list, y_list = [], []
    for i_gen in range(n_gen):
        print(f'Task Set: {i_gen + 1}/{n_gen}', end='\n')

        env.reset(solve=True)   # generates new scheduling problem

        # Optimal schedule
        # TODO: train using complete tree info, not just B&B solution?
        # TODO: generate sample weights to prioritize earliest task selections??

        t_ex, ch_ex = env.solution.t_ex, env.solution.ch_ex
        seq = np.argsort(t_ex)  # FIXME: Confirm that argsort recovers the correct sequence-to-schedule seq?!?!

        # Generate samples for each scheduling step of the optimal sequence
        for n in seq:
            n = env.sorted_index.tolist().index(n)  # transform index using sorting function

            x_list.append(env.state.copy())
            y_list.append(n)

            env.step(n)     # updates environment state

    x_set, y_set = np.array(x_list), np.array(y_list)

    # if save:
    #     d_set = tf.data.Dataset.from_tensor_slices((x_set, y_set))

    # TODO: yield?
    # TODO: partition data by tasking problem
    return x_set, y_set


def train_agent(problem_gen, n_gen_train=0, n_gen_val=0, env_cls=StepTaskingEnv, env_params=None,
                agent=None, save=False, save_dir=None):
    """
    Train a reinforcement learning agent.

    Parameters
    ----------
    problem_gen : generators.scheduling_problems.Base
        Scheduling problem generation object.
    n_gen_train : int
        Number of tasking problems to generate for agent training.
    n_gen_val : int
        Number of tasking problems to generate for agent validation.
    env_cls : class
        Gym environment class.
    env_params : dict, optional
        Parameters for environment initialization.
    agent : object
        Reinforcement learning agent.
    save : bool
        If True, the agent and environment are serialized.
    save_dir : str, optional
        String representation of sub-directory to save to.

    Returns
    -------
    function
        Wrapped agent. Takes tasks and channel availabilities and produces task execution times/channels.

    """

    if env_params is None:
        env_params = {}

    # Create environment
    env = env_cls(problem_gen, **env_params)

    if agent is None:
        agent = RandomAgent(env.infer_action_space)

    # Generate state-action data pairs
    d_train = data_gen(env, n_gen_train)
    d_val = data_gen(env, n_gen_val)

    # FIXME: load existing learning data, then split?

    # Save agent and environment
    if save:
        if save_dir is None:
            save_dir = 'temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))

        with open('agents/' + save_dir, 'wb') as file:
            dill.dump({'env': env, 'agent': agent}, file)    # save environment

    return wrap_agent(env, agent)


def load_agent(load_dir):
    """Loads agent and environment, returns wrapped scheduling function."""
    with open('agents/' + load_dir, 'rb') as file:
        pkl_dict = dill.load(file)
    return wrap_agent(**pkl_dict)


def wrap_agent(env, agent):
    """Generate scheduling function by running an agent on a single environment episode."""

    def scheduling_agent(tasks, ch_avail):
        observation, reward, done = env.reset(tasks, ch_avail), 0, False
        while not done:
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)

        return env.node.t_ex, env.node.ch_ex

    return scheduling_agent


def wrap_agent_run_lim(env, agent):
    """Generate scheduling function by running an agent on a single environment episode, enforcing max runtime."""

    def scheduling_agent(tasks, ch_avail, max_runtime):

        t_run = time.perf_counter()

        observation, reward, done = env.reset(tasks, ch_avail), 0, False
        while not done:
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)

        runtime = time.perf_counter() - t_run
        if runtime >= max_runtime:
            raise RuntimeError(f"Algorithm timeout: {runtime} > {max_runtime}.")

        return env.node.t_ex, env.node.ch_ex

    return scheduling_agent


def main():

    n_tasks = 8
    n_ch = 2

    problem_gen = RandomProblem.relu_drop_default(n_tasks, n_ch)

    #
    features = np.array([('duration', lambda task: task.duration, problem_gen.task_gen.param_lims['duration']),
                         ('release time', lambda task: task.t_release,
                          (0., problem_gen.task_gen.param_lims['t_release'][1])),
                         ('slope', lambda task: task.slope, problem_gen.task_gen.param_lims['slope']),
                         ('drop time', lambda task: task.t_drop, (0., problem_gen.task_gen.param_lims['t_drop'][1])),
                         ('drop loss', lambda task: task.l_drop, (0., problem_gen.task_gen.param_lims['l_drop'][1])),
                         ('is available', lambda task: 1 if task.t_release == 0. else 0, (0, 1)),
                         ('is dropped', lambda task: 1 if task.l_drop == 0. else 0, (0, 1)),
                         ],
                        dtype=[('name', '<U16'), ('func', object), ('lims', np.float, 2)])

    # def seq_encoding(self, n):
    #     return [0] if n in self.node.seq else [1]

    # def seq_encoding(self, n):
    #     out = np.zeros(self.n_tasks)
    #     if n in self.node.seq:
    #         out[self.node.seq.index(n)] = 1
    #     return out

    seq_encoding = 'one-hot'
    # seq_encoding = None

    def sort_func(self, n):
        if n in self.node.seq:
            return float('inf')
        else:
            return self.tasks[n].t_release
            # return 1 if self.tasks[n].l_drop == 0. else 0
            # return self.tasks[n].l_drop / self.tasks[n].t_drop

    # sort_func = 't_release'

    env_params = {'node_cls': TreeNodeShift,
                  'features': features,
                  'sort_func': sort_func,
                  'seq_encoding': seq_encoding,
                  'masking': False
                  }

    # env = SeqTaskingEnv(problem_gen, **env_params)
    env = StepTaskingEnv(problem_gen, **env_params)
    agent = RandomAgent(env.infer_action_space)

    # data_gen(env, n_gen=10)

    observation, reward, done = env.reset(), 0, False
    while not done:
        print(observation)
        print(env.sorted_index)
        print(env.node.seq)
        print(env.tasks)
        act = agent.act(observation, reward, done)
        observation, reward, done, info = env.step(act)
        print(reward)

    # act = deepq.learn(task_env,
    #                   network='mlp',
    #                   lr=1e-3,
    #                   total_timesteps=100000,
    #                   buffer_size=50000,
    #                   exploration_fraction=0.1,
    #                   exploration_final_eps=0.02,
    #                   print_freq=10,
    #                   )


if __name__ == '__main__':
    main()
