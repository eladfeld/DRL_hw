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
        self.x_min = self.gym_env.observation_space.low[0]
        self.x_max = self.gym_env.observation_space.high[0]
        self.v_min = self.gym_env.observation_space.low[1]
        self.v_max = self.gym_env.observation_space.high[1]
        self.start_point = -np.pi/6
        self.this_episode_rewards = []
        self.total_rewards = []
        self.curr_step = 0
        self.max_step = 999

    def initialize_state(self):
        self.current_state = self.gym_env.reset()
        # self.current_state = np.append(self.current_state, (self.max_steps - self.step_num) / self.max_steps)
        self.done = False
        self.curr_step = 0
        if len(self.this_episode_rewards) != 0:
            self.total_rewards.append(np.sum(self.this_episode_rewards))

    def get_state(self):
        if self.current_state is None:
            raise Exception('env is not initialized')
        else:

            # normalize and pad state
            out_state = np.zeros(shape=(self.input_size,))
            out_state[0] = ((self.current_state[0] + self.x_min) / (self.x_max - self.x_min)) * 2 - 1
            out_state[1] = ((self.current_state[1] + self.v_min) / (self.v_max - self.v_min)) * 2 - 1
            return out_state


    def is_valid_action(self, action):
        return isinstance(action, float) and action >= -1 and action <= 1


    def step(self, action):
        self.curr_step += 1
        action = np.clip(action, -1, 1)
        if not self.is_valid_action(action):
            raise ValueError('action: %s is not valid in %s environment' % (action, self.name))
        new_state, reward, done, _ = self.gym_env.step([action])
        self.current_state = new_state
        self.done = done
        self.current_state = new_state
        # if reward > 0:
        #     intrinsic_reward = (self.max_step - self.curr_step) * 0.2
        # else:
        #     intrinsic_reward = min(abs(self.current_state[0] - self.start_point) * 0.5 + abs(self.current_state[1]) * 4, 0.2)
        # intrinsic_reward = 0.1 if self.current_state[0] > 0.2 else 0
        # intrinsic_reward += min(abs(self.current_state[1]) * 4, 0.1)
        intrinsic_reward = 0
        return reward, intrinsic_reward

    def render(self):
        self.gym_env.render()

    def is_done(self):
        return self.done

    def is_converge(self):
        if len(self.total_rewards) >= 50 and np.mean(self.total_rewards[-20:] >= 90):
            return True
        return False

    def is_decay(self):
        return np.sum(self.this_episode_rewards) >= 30

    def use_intrinsic_rewards(self):
        return True