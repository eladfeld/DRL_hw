from hw1.agent_interface import AgentInterface
import numpy as np


class Agent(AgentInterface):
    def __init__(self, environment, args_dict):
        super().__init__(environment, 'lookup')
        self.args = self._read_arguments(args_dict)
        self.epsilon = self.args['epsilon']
        self.epsilon_decay_factor = self.args['epsilon_decay_factor']
        self.epsilon_decay_steps = self.args['epsilon_decay_steps']
        self._check_epsilon_greedy_parameters()
        self.states = environment.get_all_states()
        self.states_indices = {str(self.states[i]): i for i in range(len(self.states))}
        self.actions = environment.get_all_actions()
        self.actions_indices = {str(self.actions[i]): i for i in range(len(self.actions))}
        self.q_lookup_table = np.zeros(shape=[len(self.states), len(self.actions)])
        self.step = 0

    def _read_arguments(self, args_dict):
        possible_args = ['epsilon', 'epsilon_decay_factor', 'epsilon_decay_steps']
        args = {}
        for key in args_dict.keys():
            if key in possible_args:
                args[key] = args_dict[key]
        return args

    def _check_epsilon_greedy_parameters(self):
        if self.epsilon < 0 or self.epsilon >= 1:
            raise ValueError('epsilon value should be between 0 to 1')
        if self.epsilon_decay_factor < 0 or self.epsilon_decay_factor > 1:
            raise ValueError('epsilon_decay_factor value should be between 0 to 1')
        if self.epsilon_decay_steps < 1 or not isinstance(self.epsilon_decay_steps, int):
            raise ValueError('epsilon_decay_steps value should be integer bigger then  1')

    def initialize_q(self):
        self.q_lookup_table = np.zeros(shape=[len(self.states), len(self.actions)])
        self.step = 0

    def get_action_by_policy(self, state):
        q_for_state = self.q_lookup_table[self.states_indices[str(state)], :]
        if not self.train_mode or np.random.uniform(0, 1) < 1 - self.epsilon:
            best_actions_indices = np.where(q_for_state == np.max(q_for_state))[0]
            action_index = best_actions_indices[np.random.randint(len(best_actions_indices))]
        else:
            action_index = np.random.randint(len(self.actions))
        return self.actions[action_index], q_for_state[action_index]

    def update_q(self, **kwargs):
        state, action, new_q = kwargs['state'], kwargs['action'], kwargs['new_q']
        self.q_lookup_table[self.states_indices[str(state)], self.actions_indices[str(action)]] = new_q
        self.step += 1
        self._update_epsilon()

    def _update_epsilon(self):
        if self.step % self.epsilon_decay_steps == 0:
            self.epsilon = self.epsilon * self.epsilon_decay_factor

    def get_all_actions(self):
        return self.actions

    def get_action_by_max(self, state):
        q_for_state = self.q_lookup_table[self.states_indices[str(state)], :]
        action_index = np.argmax(q_for_state)
        return self.actions[action_index], q_for_state[action_index]