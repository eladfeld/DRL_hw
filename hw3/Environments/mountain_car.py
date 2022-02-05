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
        self.found_the_goal_counter = 0

    def initialize_state(self):
        self.current_state = self.gym_env.reset()
        # self.current_state = np.append(self.current_state, (self.max_steps - self.step_num) / self.max_steps)
        self.done = False
        if len(self.this_episode_rewards) != 0:
            self.total_rewards.append(np.sum(self.this_episode_rewards))

    def get_state(self):
        if self.current_state is None:
            raise Exception('env is not initialized')
        else:

            # normalize and pad state
            out_state = np.zeros(shape=(self.input_size,))
            out_state[0] = ((self.current_state[0] + self.x_min) / (self.x_max - self.x_min)) * 2 - 1
            out_state[4] = ((self.current_state[1] + self.v_min) / (self.v_max - self.v_min)) * 2 - 1
            return out_state


    def is_valid_action(self, action):
        return isinstance(action, float) and action >= -1 and action <= 1


    def step(self, action):
        action = action * 2 - 1
        if not self.is_valid_action(action):
            raise ValueError('action: %s is not valid in %s environment' % (action, self.name))
        new_state, reward, done, _ = self.gym_env.step([action])
        self.current_state = new_state
        self.done = done
        self.current_state = new_state
        if reward > 0:
            self.found_the_goal_counter += 1
        # intrinsic_reward = (abs(self.current_state[1]) * 5 + max(self.current_state[0] - self.start_point, 0)) /\
        #                    (self.found_the_goal_counter * 0.5 + 1)
        intrinsic_reward = (max(self.current_state[0] - self.start_point, 0) * 3) /\
                           (self.found_the_goal_counter * 0.5 + 1)
        return reward, intrinsic_reward

    def render(self):
        self.gym_env.render()

    def is_done(self):
        return self.done

    def is_converge(self):
        if len(self.total_rewards) >= 50 and self.total_rewards[-50:] >= 90:
            return True
        return False

    def use_intrinsic_rewards(self):
        return True