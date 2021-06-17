from abc import ABC, abstractmethod

# from gym.spaces import Discrete

from task_scheduling.learning.environments import BaseTasking


class Base(ABC):
    _learn_params_default = {}

    def __init__(self, env, model, learn_params=None):
        self.model = model

        self.env = env
        # if not isinstance(self.env.action_space, Discrete):
        #     raise TypeError("Action space must be Discrete.")

        self._set_learn_params(learn_params)
        # self.learn_params = learn_params

    def __call__(self, tasks, ch_avail):
        """
        Call scheduler, produce execution times and channels.

        Parameters
        ----------
        tasks : Sequence of task_scheduling.tasks.Base
        ch_avail : Sequence of float
            Channel availability times.

        Returns
        -------
        ndarray
            Task execution times.
        ndarray
            Task execution channels.
        """

        # ensure_valid = isinstance(self.env, envs.StepTasking) and not self.env.do_valid_actions
        # # ensure_valid = False

        obs = self.env.reset(tasks=tasks, ch_avail=ch_avail)

        done = False
        while not done:
            prob = self.obs_to_prob(obs)
            # prob = np.zeros(self.env.action_space.n)

            try:
                action = prob.argmax()
                obs, reward, done, info = self.env.step(action)
            except ValueError:
                prob = self.env.mask_probability(prob)
                action = prob.argmax()
                obs, reward, done, info = self.env.step(action)

            # if ensure_valid:
            #     prob = self.env.mask_probability(prob)
            # action = prob.argmax()
            #
            # obs, reward, done, info = self.env.step(action)

        return self.env.node.t_ex, self.env.node.ch_ex

    def _set_learn_params(self, learn_params):
        # if learn_params is None:
        #     learn_params = {}
        # self.learn_params = self._learn_params_default | learn_params
        self.learn_params = self._learn_params_default.copy()
        if learn_params is not None:
            self.learn_params.update(learn_params)

    @abstractmethod
    def obs_to_prob(self, obs):
        raise NotImplementedError

    @abstractmethod
    def learn(self, n_gen_learn, verbose=0):
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def summary(self, file=None):
        print('Env: ', end='', file=file)
        # self.env.summary(file)
        self._print_env(file)
        # print('Model\n---\n```', file=file)
        print('Model:\n```', file=file)
        self._print_model(file)
        print('```', end='\n\n', file=file)

    # @abstractmethod
    # def _print_model(self, file=None):
    #     raise NotImplementedError

    def _print_env(self, file=None):
        if isinstance(self.env, BaseTasking):
            self.env.summary(file)
        else:
            print(self.env, file=file)

    def _print_model(self, file=None):
        print(self.model, file=file)
