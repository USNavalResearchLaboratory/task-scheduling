import time
from collections import namedtuple
from pathlib import Path
import dill

import numpy as np

from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import plot_results
from stable_baselines import DQN, PPO2, A2C

from task_scheduling.generators import scheduling_problems as problem_gens
from task_scheduling.tree_search import TreeNodeShift
from task_scheduling.learning import environments as envs

np.set_printoptions(precision=2)

pkg_path = Path.cwd()
log_path = pkg_path / 'logs'
agent_path = pkg_path / 'agents'


# Agents
class RandomAgent:
    """Uniformly random action selector."""
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        action_space = self.env.action_space
        # action_space = self.env.infer_action_space(obs)
        return action_space.sample(), None       # randomly selected action

    def learn(self, *args, **kwargs):
        pass

    def set_env(self, env):
        self.env = env

    def get_env(self):
        return self.env


# Schedulers
class ReinforcementLearningScheduler:
    log_dir = log_path / 'SB_train'

    # model_cls_dict = {'Random': RandomAgent, 'DQN': DQN, 'PPO2': PPO2, 'A2C': A2C}

    _default_tuple = namedtuple('ModelDefault', ['cls', 'kwargs'])
    model_defaults = {'Random': _default_tuple(RandomAgent, {}),
                      'DQN': _default_tuple(DQN, {'policy': 'MlpPolicy', 'verbose': 1}),
                      'PPO2': _default_tuple(PPO2, {}),
                      'A2C': _default_tuple(A2C, {}),
                      }

    def __init__(self, model, env=None):
        self.model = model

        self.do_monitor = True  # TODO: make init parameter?
        if env is not None:
            self.env = env  # invokes setter

    # Environment access
    @property
    def env(self):
        return self.model.get_env()

    @env.setter
    def env(self, env):
        if isinstance(env, envs.BaseTaskingEnv):
            if self.do_monitor:
                env = Monitor(env, str(self.log_dir))
            self.model.set_env(env)
        else:
            raise TypeError("Environment must be an instance of BaseTaskingEnv.")

    def __call__(self, tasks, ch_avail):
        """
        Call scheduler, produce execution times and channels.

        Parameters
        ----------
        tasks : Iterable of tasks.Generic
        ch_avail : Iterable of float
            Channel availability times.

        Returns
        -------
        ndarray
            Task execution times.
        ndarray
            Task execution channels.
        """

        # t_run = time.perf_counter()     # FIXME: integrate runtime control
        # max_runtime = float('inf')
        # runtime = time.perf_counter() - t_run
        # if runtime >= max_runtime:
        #     raise RuntimeError(f"Algorithm timeout: {runtime} > {max_runtime}.")

        obs = self.env.reset(tasks=tasks, ch_avail=ch_avail)
        done = False
        while not done:
            action, _states = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)

        return self.env.node.t_ex, self.env.node.ch_ex

    def learn(self, n_episodes=0):
        """
        Train learning model.

        Parameters
        ----------
        n_episodes : int
            Number of complete tasking episode to train on.

        """

        self.model.learn(total_timesteps=n_episodes * self.env.steps_per_episode)

        if self.do_monitor:
            plot_results([self.log_dir], num_timesteps=None, xaxis='timesteps', task_name='Training history')

    def save(self, save_path=None):
        if save_path is None:
            cls_str = self.model.__class__.__name__
            save_path = f"temp/{cls_str}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

        save_path = agent_path / save_path
        save_path.parent.mkdir(exist_ok=True)

        self.model.save(str(save_path))

        with save_path.parent.joinpath(save_path.stem + '_env').open(mode='wb') as fid:
            env_ = self.env.env     # extract base env from Monitor
            dill.dump(env_, fid)  # save environment

    @classmethod
    def load(cls, load_path, env=None, model_cls=None):
        if model_cls is None:
            cls_str = load_path.split('/')[-1].split('_')[0]
            model_cls = cls.model_defaults[cls_str].cls
        elif isinstance(model_cls, str):
            model_cls = cls.model_defaults[model_cls].cls

        load_path = agent_path / load_path
        model = model_cls.load(str(load_path))

        try:
            with load_path.parent.joinpath(load_path.stem + '_env').open(mode='rb') as fid:
                env = dill.load(fid)
        except FileNotFoundError:
            pass

        return cls(model, env)

    # @classmethod
    # def load_from_gen(cls, load_path, problem_gen, env_cls=StepTaskingEnv, env_params=None, model_cls=None):
    #     env = env_cls.from_problem_gen(problem_gen, env_params)
    #     return cls.load(load_path, env, model_cls)

    # @classmethod
    # def from_gen(cls, model, problem_gen, env_cls=StepTaskingEnv, env_params=None):
    #     env = env_cls.from_problem_gen(problem_gen, env_params)
    #     return cls(model, env)

    @classmethod
    def train_from_gen(cls, problem_gen, env_cls=envs.SeqTaskingEnv, env_params=None, model_cls=None, model_params=None,
                       n_episodes=0, save=False, save_path=None):
        """
        Create and train a reinforcement learning scheduler.

        Parameters
        ----------
        problem_gen : generators.scheduling_problems.Base
            Scheduling problem generation object.
        env_cls : class, optional
            Gym environment class.
        env_params : dict, optional
            Parameters for environment initialization.
        model_cls : class, optional
            RL model class.
        model_params : dict, optional
            RL model parameters.
        n_episodes : int
            Number of complete environment episodes used for training.
        save : bool, optional
            If True, the agent and environment are serialized.
        save_path : str, optional
            String representation of sub-directory to save to.

        Returns
        -------
        ReinforcementLearningScheduler

        """

        env = env_cls.from_problem_gen(problem_gen, env_params)

        if model_params is None:
            model_params = {}

        if model_cls is None:
            model_cls, model_params = cls.model_defaults['Random']
        elif isinstance(model_cls, str):
            # model_cls = cls.model_cls_dict[model_cls]
            model_cls, model_params_ = cls.model_defaults[model_cls]
            model_params_.update(model_params)
            model_params = model_params_

        model = model_cls(env=env, **model_params)

        scheduler = cls(model, env)
        scheduler.learn(n_episodes)
        if save:
            scheduler.save(save_path)

        return scheduler


def main():
    # problem_gen = problem_gens.Random.relu_drop(n_tasks=4, n_ch=2)
    problem_gen = problem_gens.DeterministicTasks.relu_drop(n_tasks=4, n_ch=2)

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

    # seq_encoding = 'binary'
    # seq_encoding = None

    def sort_func(self, n):
        if n in self.node.seq:
            return float('inf')
        else:
            return self.tasks[n].t_release
            # return 1 if self.tasks[n].l_drop == 0. else 0
            # return self.tasks[n].l_drop / self.tasks[n].t_drop

    # sort_func = 't_release'

    env_cls = envs.SeqTaskingEnv
    # env_cls = envs.StepTaskingEnv

    env_params = {'node_cls': TreeNodeShift,
                  'features': features,
                  'sort_func': sort_func,
                  'masking': False,
                  'action_type': 'int',
                  # 'seq_encoding': seq_encoding,
                  }

    env = env_cls(problem_gen, **env_params)

    s = ReinforcementLearningScheduler.train_from_gen(problem_gen, env_cls, env_params,
                                                      model_cls=DQN, model_params={'policy': 'MlpPolicy', 'verbose': 1},
                                                      n_episodes=10000, save=False, save_path=None)


    # model = RandomAgent(env)
    # model = DQN('MlpPolicy', env, verbose=1)
    # model.learn(10)

    # obs = env.reset()
    # done = False
    # while not done:
    #     print(obs)
    #     # print(env.sorted_index)
    #     # print(env.node.seq)
    #     # print(env.tasks)
    #     action, _states = model.predict(obs)
    #     print(action)
    #     obs, reward, done, info = env.step(action)
    #     print(reward)


if __name__ == '__main__':
    main()
