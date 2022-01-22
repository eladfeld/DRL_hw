from hw3.environment_interface import EnvironmentInterface
from abc import ABC, abstractmethod


class AgentInterface(ABC):
    def __init__(self, environment, name='interface'):
        if not isinstance(environment, EnvironmentInterface):
            raise ValueError('environment argument should be instance of EnvironmentInterface')
        self.name = name
        self.train_mode = True

    def __str__(self):
        return self.name

    @abstractmethod
    def get_actions(self, state):
        pass