from hw3.environment_interface import EnvironmentInterface
import gym
import numpy as np

class Environment(EnvironmentInterface):
    def __init__(self, args):
        super().__init__('mountain_car')
        self.gym_env = gym.make('MountainCarContinuous-v0')
        self.current_state = None
        self.done = False
        self.step_num = 0
        self.max_steps = 200
        self.input_size = 6
        self.valid_actions = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]

    def initialize_state(self):
        self.current_state = self.gym_env.reset()
        # self.current_state = np.append(self.current_state, (self.max_steps - self.step_num) / self.max_steps)
        self.done = False
        self.step_num = 0

    def get_state(self):
        if self.current_state is None:
            raise Exception('env is not initialized')
        else:
            return np.pad(self.current_state, pad_width=(0, self.input_size - len(self.current_state)), mode='constant',
                          constant_values=0)


    def is_valid_action(self, action):
        return isinstance(action, int) and action >= 0 and action < len(self.valid_actions)


    def get_action_state_size(self):
        return len(self.valid_actions)

    def step(self, action):
        self.step_num += 1
        if not self.is_valid_action(action):
            raise ValueError('action: %s is not valid in %s environment' % (action, self.name))
        new_state, reward, done, _ = self.gym_env.step([self.valid_actions[action]])
        self.current_state = new_state
        self.done = done
        self.current_state = new_state
        return reward

    def render(self):
        self.gym_env.render()

    def is_done(self):
        return self.done
