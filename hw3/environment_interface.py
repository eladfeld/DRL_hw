from abc import ABC, abstractmethod


class EnvironmentInterface(ABC):
    def __init__(self, name='interface'):
        self.name = name
    def __str__(self):
        return self.name

    @abstractmethod
    def get_action_state_size(self):
        pass

    @abstractmethod
    def initialize_state(self):
        raise NotImplementedError('initialize_state dose not implemented by environment: %s' % self.name)

    @abstractmethod
    def get_state(self):
        raise NotImplementedError('get_state dose not implemented by environment: %s' % self.name)


    @abstractmethod
    def is_valid_action(self, action):
        raise NotImplementedError('is_valid_action dose not implemented by environment: %s' % self.name)

    @abstractmethod
    def step(self, action):
        raise NotImplementedError('step dose not implemented by environment: %s' % self.name)

    @abstractmethod
    def render(self):
        raise NotImplementedError('render dose not implemented by environment: %s' % self.name)

    @abstractmethod
    def is_done(self):
        raise NotImplementedError('is_done dose not implemented by environment: %s' % self.name)
