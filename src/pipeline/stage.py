from abc import ABC, abstractmethod


class Stage(ABC):
    @abstractmethod
    def apply(self, data):
        pass
