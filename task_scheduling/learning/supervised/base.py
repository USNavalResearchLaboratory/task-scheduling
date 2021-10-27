from abc import abstractmethod

from task_scheduling.learning.base import Base as BaseLearningScheduler


class Base(BaseLearningScheduler):
    @abstractmethod
    def predict(self, obs):
        raise NotImplementedError

    @abstractmethod
    def learn(self, n_gen_learn, verbose=0):
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError
