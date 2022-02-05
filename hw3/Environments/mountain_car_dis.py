from hw3.environment_interface import EnvironmentInterface
import gym
import numpy as np

class Environment(EnvironmentInterface):
    def __init__(self, args):
        super().__init__('mountain_car')
        self.gym_env = gym.make('MountainCarContinuous-v0')
        self.current_state = None
        self.done = False
        self.input_size = 6
        self.valid_actions = [-1, 1]

        self.v_min = self.gym_env.observation_space.low[1]
        self.v_max = self.gym_env.observation_space.high[1]
        self.episode_rewards = []
        self.total_rewards = []
        self.start_point = -np.pi / 6

    def initialize_state(self):
        if len(self.episode_rewards) > 0:
            self.total_rewards.append(np.sum(self.episode_rewards))
            self.episode_rewards = []
        self.current_state = self.gym_env.reset()
        self.done = False

    def get_state(self):
        if self.current_state is None:
            raise Exception('env is not initialized')
        else:

            # normalize and pad state
            out_state = np.zeros(shape=(self.input_size,))
            out_state[0] = self.current_state[0]
            out_state[2] = ((self.current_state[1] + self.v_min) / (self.v_max - self.v_min)) * 2 - 1
            return out_state


    def is_valid_action(self, action):
        return isinstance(action, int) and action >= 0 and action <= len(self.valid_actions)


    def step(self, action):
        if not self.is_valid_action(action):
            raise ValueError('action: %s is not valid in %s environment' % (action, self.name))
        action = self.valid_actions[action]
        new_state, reward, done, _ = self.gym_env.step([action])
        self.current_state = new_state
        self.done = done
        self.current_state = new_state
        intrinsic_reward = 0
        if reward > 0:
            intrinsic_reward = 1 - reward
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
        return True
