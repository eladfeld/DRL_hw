from environment_interface import EnvironmentInterface
from abc import ABC, abstractmethod


class AgentInterface(ABC):
    def __init__(self, environment, name='interface'):
        if not isinstance(environment, EnvironmentInterface):
            raise ValueError('environment argument should be instance of EnvironmentInterface')
        self.name = name
        self.train_mode = True

    def __str__(self):
        return self.name
    def set_train_mode(self, mode=True):
        self.train_mode = mode

    @abstractmethod
    def initialize_q(self):
        raise NotImplementedError('initialize_q dose not implemented by agent: %s' % self.name)

    @abstractmethod
    def get_action_by_policy(self, state):
        raise NotImplementedError('get_action_by_policy dose not implemented by agent: %s' % self.name)

    @abstractmethod
    def update_q(self, state, action, new_q):
        raise NotImplementedError('update_q dose not implemented by agent: %s' % self.name)

    @abstractmethod
    def get_all_actions(self):
        raise NotImplementedError('get_all_actions dose not implemented by agent: %s' % self.name)

    @abstractmethod
    def get_action_by_max(self, state):
        raise NotImplementedError('get_action_by_max dose not implemented by agent: %s' % self.name)