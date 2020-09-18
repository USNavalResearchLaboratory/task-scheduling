import time
import os
from copy import deepcopy
from types import MethodType
from math import factorial
import dill

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.spaces import Box, Space, Discrete

from util.plot import plot_task_losses
from util.generic import seq2num, num2seq
from generators.scheduling_problems import Random as RandomProblem
from tree_search import TreeNode, TreeNodeShift

############################ new includes ###########################
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as MlpPolicyQ
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, A2C
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter
from stable_baselines.common.callbacks import BaseCallback

#####################################################################

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


###############################################################
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


###############################################################

# Gym Spaces
class Sequence(Space):
    """Gym Space for index sequences."""

    def __init__(self, n):
        self.n = n      # sequence length
        super().__init__((self.n,), np.int)

    def sample(self):
        return self.np_random.permutation(self.n)

    def contains(self, x):
        return True if (np.sort(np.asarray(x, dtype=int)) == np.arange(self.n)).all() else False

    def __repr__(self):
        return f"Sequence({self.n})"

    def __eq__(self, other):
        return isinstance(other, Sequence) and self.n == other.n

    def __len__(self):
        return factorial(self.n)


class DiscreteSet(Space):
    """Gym Space for discrete, non-integral elements."""

    def __init__(self, elements):
        self.elements = np.sort(np.array(list(elements)).flatten())   # ndarray representation of set
        super().__init__((), self.elements.dtype)

    def sample(self):
        return self.np_random.choice(self.elements)

    def contains(self, x):
        return True if x in self.elements else False

    def __repr__(self):
        return f"DiscreteSet({self.elements})"

    def __eq__(self, other):
        return isinstance(other, DiscreteSet) and self.elements == other.elements

    def __len__(self):
        return self.elements.size


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
    masking : bool
        If True, features are zeroed out for scheduled tasks.

    """

    def __init__(self, problem_gen, node_cls=TreeNode, features=None, sort_func=None, masking=False):
        self.problem_gen = problem_gen
        self.solution = None

        self.n_tasks = self.problem_gen.n_tasks
        self.n_ch = self.problem_gen.n_ch

        # Set features and state bounds
        if features is not None:
            self.features = features
        else:
            self.features = self.problem_gen.task_gen.default_features

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

        self.masking = masking

        self.reward_range = (-float('inf'), 0)
        self.loss_agg = None

        self.node_cls = node_cls
        self.node = None

    tasks = property(lambda self: self.node.tasks)
    ch_avail = property(lambda self: self.node.ch_avail)

    def __repr__(self):
        if self.node is None:
            _stat = 'Initialized'
        else:
            _stat = f'{len(self.node.seq)}/{self.n_tasks} Tasks Scheduled'
        return f"{self.__class__.__name__}({_stat})"

    @property
    def sorted_index(self):
        """Indices for task re-ordering for environment state."""
        if callable(self.sort_func):
            return np.argsort([self.sort_func(n) for n in range(self.n_tasks)])
        else:
            return np.arange(self.n_tasks)

    @property
    def sorted_index_rev(self):
        return np.array([self.sorted_index.tolist().index(n) for n in range(self.n_tasks)])
        # return np.flatnonzero(np.isin(self.sorted_index, np.arange(self.n_tasks)))

    @property
    def state_tasks(self):
        """State sub-array for task features."""
        state_tasks = np.array([task.feature_gen(*self.features['func']) for task in self.tasks])
        if self.masking:
            state_tasks[self.node.seq] = 0.     # zero out state rows for scheduled tasks

        return state_tasks[self.sorted_index]       # sort individual task states

    @property
    def state(self):
        """Complete state."""
        return self.state_tasks

    def _update_spaces(self):
        """Update observation and action spaces."""
        pass

    def reset(self, tasks=None, ch_avail=None, persist=False, solve=False, verbose=False):
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
        verbose : bool, optional
            Enables print-out progress information.

        """

        if not persist:
            if tasks is None or ch_avail is None:   # generate new scheduling problem
                if solve:   # TODO: next()? Pass a generator, not a callable??
                    ((tasks, ch_avail), self.solution), = self.problem_gen(1, solve=solve, verbose=verbose)
                else:
                    (tasks, ch_avail), = self.problem_gen(1, solve=solve, verbose=verbose)
                    self.solution = None

            elif len(tasks) != self.n_tasks:
                raise ValueError(f"Input 'tasks' must be None or a list of {self.n_tasks} tasks")
            elif len(ch_avail) != self.n_ch:
                raise ValueError(f"Input 'ch_avail' must be None or an array of {self.n_ch} channel availabilities")

            self.node_cls._tasks_init = tasks
            self.node_cls._ch_avail_init = ch_avail

        self.node = self.node_cls()
        self.loss_agg = self.node.l_ex  # Loss can be non-zero due to time origin shift during node initialization

        self._update_spaces()

        return self.state

    def step(self, action):
        """
        Updates environment state based on task index input.

        Parameters
        ----------
        action : int or Sequence of int
            Complete index sequence.

        Returns
        -------
        observation : ndarray
        reward : float
            Negative loss achieved by the complete sequence.
        done : True
            Episode completes after one step.
        info : dict
            Auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        """

        action = self.sorted_index[action]  # decode task index to original order
        self.node.seq_extend(action)  # updates sequence, loss, task parameters, etc.

        reward, self.loss_agg = self.loss_agg - self.node.l_ex, self.node.l_ex
        done = len(self.node.seq_rem) == 0      # sequence is complete

        self._update_spaces()

        return self.state, reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            fig_env, ax_env = plt.subplots(num='Task Scheduling Env', clear=True)
            plot_task_losses(self.tasks, ax=ax_env)

    def close(self):
        plt.close('all')

    def data_gen(self, n_batch, batch_size=1, weight_func=None, verbose=False):
        """
        Generate state-action data for learner training and evaluation.

        Parameters
        ----------
        n_batch : int
            Number of batches of state-action pair data to generate.
        batch_size : int
            Number of scheduling problems to make data from per yielded batch.
        weight_func : callable, optional
            Function mapping partial sequence length and number of tasks to a training weight.
        verbose : bool, optional
            Enables print-out progress information.

        Yields
        ------
        ndarray
            Predictor data.
        ndarray
            Target data.
        ndarray, optional
            Sample weights.

        """

        for i_batch in range(n_batch):
            if verbose:
                print(f'Batch: {i_batch + 1}/{n_batch}', end='\n')

            x_set, y_set, w_set = [], [], []
            for i_gen in range(batch_size):
                self.reset(solve=True, verbose=verbose)  # generates new scheduling problem

                # Optimal schedule
                t_ex, ch_ex = self.solution.t_ex, self.solution.ch_ex
                seq = np.argsort(t_ex)  # maps to optimal schedule (empirically supported...)

                # TODO: train using complete tree info, not just B&B solution?

                # Generate samples for each scheduling step of the optimal sequence
                x_set_single, y_set_single, w_set_single = self._gen_single(seq, weight_func)
                x_set.extend(x_set_single)
                y_set.extend(y_set_single)
                w_set.extend(w_set_single)

            if callable(weight_func):
                yield np.array(x_set), np.array(y_set), np.array(w_set)
            else:
                yield np.array(x_set), np.array(y_set)

    def _gen_single(self, seq, weight_func):
        """Generate lists of predictor/target/weight samples for a given optimal task index sequence."""
        raise NotImplementedError


class SeqTaskingEnv(BaseTaskingEnv):
    """Tasking environment with single action of a complete task index sequence."""

    def __init__(self, problem_gen, node_cls=TreeNode, features=None, sort_func=None, masking=False,
                 action_type='seq'):
        super().__init__(problem_gen, node_cls, features, sort_func, masking)

        self.action_type = action_type      # 'seq' for Sequences, 'int' for Integers
        if self.action_type == 'seq':
            self.action_space_map = lambda n: Sequence(n)
        elif self.action_type == 'int':
            self.action_space_map = lambda n: Discrete(factorial(n))
        else:
            raise ValueError

        # gym.Env observation and action spaces
        self.observation_space = Box(self._state_tasks_low, self._state_tasks_high, dtype=np.float64)
        self.action_space = self.action_space_map(self.n_tasks)

    def infer_action_space(self, observation):
        """Determines the action Gym.Space from an observation."""
        return self.action_space_map(len(observation))

    def step(self, action):
        if self.action_type == 'seq':
            pass
        elif self.action_type == 'int':
            action = list(num2seq(action, self.n_tasks))
        else:
            raise ValueError

        return super().step(action)

    def _gen_single(self, seq, weight_func):
        """Generate lists of predictor/target/weight samples for a given optimal task index sequence."""
        seq_sort = self.sorted_index_rev[seq]

        x_set = [self.state.copy()]

        if self.action_type == 'seq':
            y_set = [seq_sort]
        elif self.action_type == 'int':
            y_set = [seq2num(seq_sort)]
        else:
            raise ValueError

        if callable(weight_func):
            w_set = [weight_func()]  # TODO: weighting based on loss value? Use MethodType, or new call signature?
        else:
            w_set = []

        self.step(seq_sort)

        return x_set, y_set, w_set


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
    masking : bool
        If True, features are zeroed out for scheduled tasks.
    seq_encoding : function or str, optional
        Method that returns a 1-D encoded sequence representation for a given task index 'n'. Assumes that the
        encoded array sums to one for scheduled tasks and to zero for unscheduled tasks.

    """

    def __init__(self, problem_gen, node_cls=TreeNode, features=None, sort_func=None, masking=False,
                 seq_encoding='one-hot'):

        super().__init__(problem_gen, node_cls, features, sort_func, masking)

        # Set sequence encoder method
        if callable(seq_encoding):
            self.seq_encoding = MethodType(seq_encoding, self)

            env_copy = deepcopy(self)       # FIXME: hacked - find better way!
            env_copy.reset()
            self.len_seq_encode = env_copy.state_seq.shape[-1]
        elif type(seq_encoding) == str:     # simple string specification for supported encoders
            if seq_encoding == 'indicator':
                def _seq_encoding(env, n):
                    return [1] if n in env.node.seq else [0]

                self.len_seq_encode = 1
            elif seq_encoding == 'one-hot':
                def _seq_encoding(env, n):
                    out = np.zeros(env.n_tasks)
                    if n in env.node.seq:
                        out[env.node.seq.index(n)] = 1
                    return out

                self.len_seq_encode = self.n_tasks
            else:
                raise ValueError("Unsupported sequence encoder string.")

            self.seq_encoding = MethodType(_seq_encoding, self)
        else:
            raise TypeError("Sequence encoding input must be callable or str.")

        # gym.Env observation and action spaces
        _state_low = np.concatenate((np.zeros((self.n_tasks, self.len_seq_encode)), self._state_tasks_low), axis=1)
        _state_high = np.concatenate((np.ones((self.n_tasks, self.len_seq_encode)), self._state_tasks_high), axis=1)
        self.observation_space = Box(_state_low, _state_high, dtype=np.float64)
        self.action_space = DiscreteSet(set(range(self.n_tasks)))

    def infer_action_space(self, observation):
        """Determines the action Gym.Space from an observation."""
        _state_seq = observation[:, :-len(self.features)]
        return DiscreteSet(np.flatnonzero(1 - _state_seq.sum(1)))

    def _update_spaces(self):
        """Update observation and action spaces."""
        seq_rem_sort = self.sorted_index_rev[list(self.node.seq_rem)]
        self.action_space = DiscreteSet(seq_rem_sort)

    @property
    def state_seq(self):
        """State sub-array for encoded partial sequence."""
        state_seq = np.array([self.seq_encoding(n) for n in range(self.n_tasks)])
        return state_seq[self.sorted_index]  # sort individual sequence states

    @property
    def state(self):
        """Complete state."""
        return np.concatenate((self.state_seq, self.state_tasks), axis=1)

    def _gen_single(self, seq, weight_func):
        """Generate lists of predictor/target/weight samples for a given optimal task index sequence."""
        x_set, y_set, w_set = [], [], []

        for i, n in enumerate(seq):
            n = self.sorted_index_rev[n]

            x_set.append(self.state.copy())
            y_set.append(n)
            if callable(weight_func):
                w_set.append(weight_func(i, self.n_tasks))

            self.step(n)  # updates environment state

        return x_set, y_set, w_set


# Agents
class RandomAgent:
    """The world's simplest agent!"""
    def __init__(self, infer_action_space):
        self.infer_action_space = infer_action_space

    def act(self, observation, reward, done):
        action_space = self.infer_action_space(observation)
        return action_space.sample()       # randomly selected action


# Learning
def train_agent(problem_gen, n_batch_train=1, n_batch_val=1, batch_size=1, env_cls=StepTaskingEnv, env_params=None,
                agent=None, save=False, save_dir=None):
    """
    Train a reinforcement learning agent.

    Parameters
    ----------
    problem_gen : generators.scheduling_problems.Base
        Scheduling problem generation object.
    n_batch_train : int
        Number of batches of state-action pair data to generate for agent training.
    n_batch_val : int
        Number of batches of state-action pair data to generate for agent validation.
    batch_size : int
        Number of scheduling problems to make data from per yielded batch.
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

    # Generate state-action data pairs, train
    # TODO

    # Save agent and environment
    if save:
        if save_dir is None:
            save_dir = 'temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))

        with open('../agents/' + save_dir, 'wb') as file:
            dill.dump({'env': env, 'agent': agent}, file)    # save environment

    return wrap_agent(env, agent)

###########################################################################################
def evaluate(model, env, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action)

        # Stats
        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward


################################ new function ##############################################
def train_agent_sb(problem_gen, n_batch_train=1, n_batch_val=1, batch_size=1, env_cls=SeqTaskingEnv, env_params=None,
                agent=None, save=False, save_dir=None, algorithm='DQN', policy='mlp', timesteps=1000):
    if env_params is None:
        env_params = {}

    # Create environment
    env = env_cls(problem_gen, **env_params)

    # Create log dir
    log_dir = "../tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)

    if agent is None:
        agent = RandomAgent(env.infer_action_space)

    if (algorithm == 'DQN') and (policy == 'mlp'): 
        agent = DQN(MlpPolicyQ, env)
    elif (algorithm == 'PPO2') and (policy == 'mlp'):
        agent = PPO2(MlpPolicy, env)
    elif (algorithm == 'A2C') and (policy == 'mlp'):
        agent = A2C(MlpPolicy, env)
        env = DummyVecEnv([lambda: env])

    else:
        raise NotImplementedError
    print ('training stable baselines agent . . . good luck')
    mean_reward_before_train = evaluate(agent, env, num_steps=10000)



    if save:
        # checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='../agents/',
        #                                  name_prefix=algorithm + policy)
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

        agent.learn(total_timesteps=timesteps, callback=callback)
        # agent.learn(total_timesteps=timesteps, callback=checkpoint_callback, log_interval=100, tb_log_name="DQN")

    else:
        agent.learn(total_timesteps=timesteps)
    # Save agent and environment
#    if save:
#        if save_dir is None:
#            save_dir = 'temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))

#        with open('../agents/' + save_dir, 'wb') as file:
#            dill.dump({'env': env, 'agent': agent}, file)    # save environment
    plot_results(log_dir)
    mean_reward = evaluate(agent, env, num_steps=10000)

    return wrap_agent_sb(env, agent)
###########################################################################################

def load_agent(load_dir):
    """Loads agent and environment, returns wrapped scheduling function."""
    with open('../agents/' + load_dir, 'rb') as file:
        pkl_dict = dill.load(file)
    return wrap_agent(**pkl_dict)

################################ new function ##############################################
def load_agent_sb(load_dir, env):
    """Loads agent and environment, returns wrapped scheduling function."""
#    with open('../agents/' + load_dir, 'rb') as file:
#        pkl_dict = dill.load(file)
    if ('DQN' in load_dir) and ('mlp' in load_dir):
        agent = DQN(MlpPolicyQ, env, verbose=1)
        agent.load(load_dir, env)
    else:
        raise NotImplementedError

    return wrap_agent_sb(env, agent)

##########################################################################################

def wrap_agent(env, agent):
    """Generate scheduling function by running an agent on a single environment episode."""

    def scheduling_agent(tasks, ch_avail):
        observation, reward, done = env.reset(tasks, ch_avail), 0, False
        while not done:
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)

        return env.node.t_ex, env.node.ch_ex

    return scheduling_agent
################################ new function ##############################################
def wrap_agent_sb(env, agent):
    """Generate scheduling function by running an agent on a single environment episode."""

    def scheduling_agent(tasks, ch_avail):
        observation, reward, done = env.reset(tasks, ch_avail), 0, False
        while not done:
            action, _ = agent.predict(observation, reward, done)
            observation, reward, done, info = env.step(action)

        return env.node.t_ex, env.node.ch_ex

    return scheduling_agent
###########################################################################################

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

    problem_gen = RandomProblem.relu_drop_default(n_tasks=6, n_ch=2)

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
    # features = None

    # def seq_encoding(self, n):
    #     return [0] if n in self.node.seq else [1]

    def seq_encoding(self, n):
        out = np.zeros(self.n_tasks)
        if n in self.node.seq:
            out[self.node.seq.index(n)] = 1
        return out

    # seq_encoding = 'indicator'
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
                  'masking': False,
                  'action_type': 'int',
                  # 'seq_encoding': seq_encoding,
                  }
    env = SeqTaskingEnv(problem_gen, **env_params)
    # env = StepTaskingEnv(problem_gen, **env_params)
    agent = RandomAgent(env.infer_action_space)

#    agent = DQN(MlpPolicyQ, env, verbose=1)
#    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='../agents/',
#                                         name_prefix='rl_model')
#    agent.learn(total_timesteps=1000,callback=checkpoint_callback)
#    del agent 
#    agent = DQN(MlpPolicyQ, env, verbose=1)
#    agent.load('../agents/rl_model_1000_steps.zip', env)

    for i in range(10):
    # out = list(env.data_gen(3, batch_size=2, verbose=True))

        observation, reward, done = env.reset(), 0, False
        while not done:
            print(observation)
            print(env.sorted_index)
            print(env.node.seq)
            print(env.tasks)
#            act, _ = agent.predict(observation, reward, done)
#            act = agent.learn(total_timesteps=1000)
            act = agent.act(observation,reward,done)
            observation, reward, done, info = env.step(act)


if __name__ == '__main__':
    main()
