import time

import numpy as np

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

from generators.scheduling_problems import Random as RandomProblem
from tree_search import TreeNode, TreeNodeShift
from environments import SeqTaskingEnv, StepTaskingEnv

np.set_printoptions(precision=2)


# Agents
class RandomAgent:      # TODO: implement as BaseRLModel subclass for common API?
    """Uniformly random action selector."""
    def __init__(self, env):
        self.env = env

    def learn(self, *args, **kwargs):
        pass

    def predict(self, obs):
        # action_space = self.env.action_space
        action_space = self.env.infer_action_space(obs)
        return action_space.sample(), None       # randomly selected action


# Learning
def train_agent(problem_gen, n_episodes=0, env_cls=SeqTaskingEnv, env_params=None,
                agent=None, save=False, save_dir=None):     # FIXME
    """
    Train a reinforcement learning agent.

    Parameters
    ----------
    problem_gen : generators.scheduling_problems.Base
        Scheduling problem generation object.
    n_episodes : int
        Number of complete environment episodes used for training.
    env_cls : class, optional
        Gym environment class.
    env_params : dict, optional
        Parameters for environment initialization.
    agent : BaseRLModel or str, optional
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

    # Create environment
    if env_params is None:
        env_params = {}

    env = env_cls(problem_gen, **env_params)

    # Train agent
    if agent is None or agent == 'random':
        agent = RandomAgent(env)
        # agent = RandomAgent(env.infer_action_space)
    elif agent == 'DQN':
        agent = DQN(MlpPolicy, env, verbose=1)

    agent.learn(total_timesteps=n_episodes*env.steps_per_episode)

    # # Save agent and environment  # FIXME
    # if save:
    #     if save_dir is None:
    #         save_dir = 'temp/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))
    #
    #     with open('../agents/' + save_dir, 'wb') as file:
    #         dill.dump({'env': env, 'agent': agent}, file)    # save environment

    return wrap_agent(env, agent)


# def load_agent(load_dir):     # FIXME
#     """Loads agent and environment, returns wrapped scheduling function."""
#     with open('../agents/' + load_dir, 'rb') as file:
#         pkl_dict = dill.load(file)
#     return wrap_agent(**pkl_dict)


def wrap_agent(env, agent):     # TODO: refactor as Env methods? Consider stable_baselines save/load...
    """Generate scheduling function by running an agent on a single environment episode."""

    def scheduling_agent(tasks, ch_avail):
        obs = env.reset(tasks, ch_avail)
        done = False
        while not done:
            action, _states = agent.predict(obs)        # TODO: use SB action probs?
            obs, reward, done, info = env.step(action)

        return env.node.t_ex, env.node.ch_ex

    return scheduling_agent


def wrap_agent_run_lim(env, agent):
    """Generate scheduling function by running an agent on a single environment episode, enforcing max runtime."""

    def scheduling_agent(tasks, ch_avail, max_runtime):

        t_run = time.perf_counter()

        obs = env.reset(tasks, ch_avail)
        done = False
        while not done:
            action, _states = agent.predict(obs)
            obs, reward, done, info = env.step(action)

        runtime = time.perf_counter() - t_run
        if runtime >= max_runtime:
            raise RuntimeError(f"Algorithm timeout: {runtime} > {max_runtime}.")

        return env.node.t_ex, env.node.ch_ex

    return scheduling_agent


def main():
    problem_gen = RandomProblem.relu_drop_default(n_tasks=8, n_ch=2)

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

    # env_cls = SeqTaskingEnv
    env_cls = StepTaskingEnv

    env_params = {'node_cls': TreeNodeShift,
                  'features': features,
                  'sort_func': sort_func,
                  'masking': False,
                  # 'action_type': 'int',
                  'seq_encoding': seq_encoding,
                  }

    env = env_cls(problem_gen, **env_params)

    # out = list(env.data_gen(5, batch_size=2, verbose=True))

    agent = RandomAgent(env)
    # agent = DQN(MlpPolicy, env, verbose=1)

    agent.learn(10)

    # scheduler = train_agent(problem_gen, n_episodes=10, env_cls=env_cls, env_params=env_params, agent=agent)

    obs = env.reset()
    done = False
    while not done:
        print(obs)
        # print(env.sorted_index)
        # print(env.node.seq)
        # print(env.tasks)
        action, _states = agent.predict(obs)
        print(action)
        obs, reward, done, info = env.step(action)
        print(reward)


if __name__ == '__main__':
    main()
