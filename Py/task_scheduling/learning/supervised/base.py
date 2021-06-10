from abc import ABC, abstractmethod


class BaseSupervisedScheduler(ABC):

    # TODO: complete base class API

    @abstractmethod
    def learn(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def summary(self, file=None):
        raise NotImplementedError
