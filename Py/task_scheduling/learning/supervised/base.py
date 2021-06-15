from abc import ABC, abstractmethod

from gym.spaces import Discrete


class Base(ABC):

    # TODO: complete base class API

    def __init__(self, env, model):
        self.env = env
        # if not isinstance(self.env.action_space, Discrete):
        #     raise TypeError("Action space must be Discrete.")

        self.model = model

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

    @abstractmethod
    def obs_to_prob(self, obs):
        raise NotImplementedError

    @abstractmethod
    def learn(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError

    # @abstractmethod
    # def summary(self, file=None):
    #     raise NotImplementedError

    def summary(self, file=None):
        print('Env: ', end='', file=file)
        self.env.summary(file)
        print('Model\n---\n```', file=file)
        self._print_model(file)
        print('```', end='\n\n', file=file)

    @abstractmethod
    def _print_model(self, file=None):
        raise NotImplementedError
