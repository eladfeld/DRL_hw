from hw1.environment_interface import EnvironmentInterface
import gym


class Environment(EnvironmentInterface):
    def __init__(self):
        super().__init__('frozen_lake')
        self.gym_env = gym.make('FrozenLake-v1')
        self.current_state = None
        self.done = False
        self.valid_actions = list(range(4))

    def initialize_state(self):
        self.gym_env.reset()
        self.current_state = 0

    def get_state(self):
        if self.current_state is None:
            raise Exception('env is not initialized')
        else:
            return self.current_state

    def get_all_states(self):
        return list(range(16))

    def is_valid_action(self, action):
        return action in self.valid_actions

    def step(self, action):
        if not self.is_valid_action(action):
            raise ValueError('action: %s is not valid in %s environment' % (action, self.name))
        new_state, reward, done, _ = self.gym_env.step(action)
        self.current_state = new_state
        self.done = done
        return reward

    def render(self):
        self.render()

    def is_done(self):
        return self.done

    def get_all_actions(self):
        return self.valid_actions
