from hw1.environment_interface import EnvironmentInterface
from hw1.agent_interface import AgentInterface
from abc import ABC, abstractmethod


class ExperimentInterface(ABC):
    def __init__(self, environment, agent, args_dict, name='interface'):
        if not isinstance(environment, EnvironmentInterface):
            raise ValueError('environment argument should be instance of EnvironmentInterface')
        if not isinstance(agent, AgentInterface):
            raise ValueError('agent argument should be instance of EnvironmentInterface')
        self.name = name

    def __str__(self):
        return self.name

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError('update dose not implemented by Experiments: %s' % self.name)

    @abstractmethod
    def show(self):
        raise NotImplementedError('show dose not implemented by Experiments: %s' % self.name)

    @abstractmethod
    def is_done(self):
        raise NotImplementedError('is_done dose not implemented by Experiments: %s' % self.name)
