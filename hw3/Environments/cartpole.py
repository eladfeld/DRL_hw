from hw3.environment_interface import EnvironmentInterface
import gym
import numpy as np

class Environment(EnvironmentInterface):
    def __init__(self, args):
        super().__init__('cartpole')
        self.gym_env = gym.make('CartPole-v1')
        self.current_state = None
        self.done = False
        self.input_size = 6
        self.valid_actions = [0, 1]
        self.episode_rewards = []
        self.total_rewards = []

    def initialize_state(self):
        if len(self.episode_rewards) > 0:
            self.total_rewards.append(np.sum(self.episode_rewards))
            self.episode_rewards = []
        self.current_state = self.gym_env.reset()
        # self.current_state = np.append(self.current_state, (self.max_steps - self.step_num) / self.max_steps)
        self.done = False

    def get_state(self):
        if self.current_state is None:
            raise Exception('env is not initialized')
        else:
            return np.pad(self.current_state, pad_width=(0, self.input_size - len(self.current_state)), mode='constant',
                          constant_values=0)


    def is_valid_action(self, action):
        return isinstance(action, int) and action >= 0 and action <= len(self.valid_actions)


    def step(self, action):
        if not self.is_valid_action(action):
            raise ValueError('action: %s is not valid in %s environment' % (action, self.name))
        action = self.valid_actions[action]
        new_state, reward, done, _ = self.gym_env.step(action)
        self.current_state = new_state
        self.done = done
        self.current_state = new_state
        intrinsic_reward = 0
        self.episode_rewards.append(reward)
        return reward, intrinsic_reward

    def render(self):
        self.gym_env.render()

    def is_done(self):
        return self.done

    def get_action_state_size(self):
        return len(self.valid_actions)

    def is_converge(self):
        if len(self.total_rewards) >= 100 and np.mean(self.total_rewards[-100:]) >= 475:
            return True
        return False

    def use_intrinsic_rewards(self):
        return False
