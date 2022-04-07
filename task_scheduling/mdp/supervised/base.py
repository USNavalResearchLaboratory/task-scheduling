"""Base classes for supervised learning."""

from abc import abstractmethod

from task_scheduling.mdp.base import BaseLearning


class Base(BaseLearning):  # TODO: deprecate? Only used for type checking??
    @abstractmethod
    def predict(self, obs):
        raise NotImplementedError

    @abstractmethod
    def learn(self, n_gen_learn, verbose=0):
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError
