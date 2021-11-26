from hw1.environment_interface import EnvironmentInterface
import gym
import numpy as np

class Environment(EnvironmentInterface):
    def __init__(self, args):
        super().__init__('cart_pole')
        self.gym_env = gym.make('CartPole-v1')
        self.current_state = None
        self.done = False
        self.valid_actions = list(range(2))
        self.step_num = 0
        self.max_steps = args['steps']
        self.discount_factor = args['discount_factor']

    def initialize_state(self):
        self.current_state = self.gym_env.reset()
        # self.current_state = np.append(self.current_state, (self.max_steps - self.step_num) / self.max_steps)
        self.done = False
        self.step_num = 0

    def get_state(self):
        if self.current_state is None:
            raise Exception('env is not initialized')
        else:
            return self.current_state

    def get_all_states(self):
        return (4, )

    def is_valid_action(self, action):
        return action in self.valid_actions

    def step(self, action):
        self.step_num += 1
        if not self.is_valid_action(action):
            raise ValueError('action: %s is not valid in %s environment' % (action, self.name))
        new_state, reward, done, _ = self.gym_env.step(action)
        # new_state = np.append(new_state, (self.max_steps - self.step_num) / self.max_steps)
        self.current_state = new_state
        self.done = done
        return reward

    def render(self):
        self.gym_env.render()

    def is_done(self):
        return self.done

    def get_all_actions(self):
        return self.valid_actions
