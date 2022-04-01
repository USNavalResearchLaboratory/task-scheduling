from abc import ABC, abstractmethod

from task_scheduling.mdp.environments import Base as BaseEnv


class Base(ABC):
    def __init__(self, env):
        """
        Base class for learning schedulers.

        Parameters
        ----------
        env : BaseEnv
            OpenAi gym environment.

        """
        if not isinstance(env, BaseEnv):
            raise TypeError(f"`env` must be of type BaseEnv, got {type(env)}")
        self.env = env

    def __call__(self, tasks, ch_avail):
        """
        Call scheduler, produce execution times and channels.

        Parameters
        ----------
        tasks : Collection of task_scheduling.tasks.Base
        ch_avail : Collection of float
            Channel availability times.

        Returns
        -------
        ndarray
            Task execution times.
        ndarray
            Task execution channels.
        """

        obs = self.env.reset(tasks=tasks, ch_avail=ch_avail)

        done = False
        while not done:
            action = self.predict(obs)
            obs, reward, done, info = self.env.step(action)

        return self.env.node.sch

    @abstractmethod
    def predict(self, obs):
        raise NotImplementedError

    def summary(self):
        out = "Env:" \
              f"\n{self._print_env()}"
        return out

    def _print_env(self):
        if isinstance(self.env, BaseEnv):
            return self.env.summary()
        else:
            return str(self.env)


class RandomAgent(Base):
    """Uniformly random action selector."""
    def predict(self, obs):
        action_space = self.env.action_space
        # action_space = self.env.infer_action_space(obs)

        return action_space.sample()
        # return action_space.sample(), None


class BaseLearning(Base):
    _learn_params_default = {}

    def __init__(self, env, model, learn_params=None):
        """
        Base class for learning schedulers.

        Parameters
        ----------
        env : BaseEnv
            OpenAi gym environment.
        model
            The learning object.
        learn_params : dict, optional
            Parameters used by the `learn` method.

        """
        self.model = model
        super().__init__(env)

        self._learn_params = self._learn_params_default.copy()
        if learn_params is None:
            learn_params = {}
        self.learn_params = learn_params  # invoke property setter

    @property
    def learn_params(self):
        return self._learn_params

    @learn_params.setter
    def learn_params(self, params):
        self._learn_params.update(params)

    @abstractmethod
    def learn(self, n_gen_learn, verbose=0):
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def summary(self):
        str_model = f"\n\n" \
                    f"Model:\n" \
                    f"{self._print_model()}"
        return super().summary() + '\n\n' + str_model

    def _print_model(self):
        return f"```\n" \
               f"{str(self.model)}\n" \
               f"```"
