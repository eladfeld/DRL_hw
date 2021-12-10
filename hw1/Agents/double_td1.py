from hw1.agent_interface import AgentInterface
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.models import clone_model, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K

class Agent(AgentInterface):
    def __init__(self, environment, args_dict):
        super().__init__(environment, 'double_td1')
        self.args = self._read_arguments(args_dict)
        self.epsilon = self.args['epsilon']
        self.epsilon_decay_factor = self.args['epsilon_decay_factor']
        self.epsilon_decay_steps = self.args['epsilon_decay_steps']
        self._check_arguments()
        self.states_shape = environment.get_all_states()
        self.actions = environment.get_all_actions()
        self.environment = environment
        self.actions_indices = {self.actions[i]: i for i in range(len(self.actions))}
        self.q_network1 = None
        self.q_network2 = None
        self.training_model1 = None
        self.training_model2 = None
        self.target_update_episodes = self.args['target_update_episodes']
        self.step = 0
        self.episode = 0
        self.discount_factor = self.args['discount_factor']
        self.learning_rate = self.args['learning_rate']
        self.min_epsilon = self.args['min_epsilon']
        self.min_lr = self.args['min_lr']
        self.lr_decay_factor = self.args['lr_decay_factor']
        self.perfect_counter = 0
        self.episode_losses = []
        self.last_episode_loss = 0
        self.plato_phase = False
        self.is_1 = True
        self.experience_replay = []
        self.experience_replay_capacity = self.args['experience_replay_capacity']
        self.batch_size = self.args['batch_size']
        self.last_step_cache = {'state':None, 'action':None, 'reward':None, 'new_state':None ,'d':None}

    def _read_arguments(self, args_dict):
        possible_args = ['epsilon', 'epsilon_decay_factor', 'epsilon_decay_steps', 'min_epsilon', 'layers',
                         'learning_rate', 'target_update_episodes', 'steps', 'discount_factor', 'lr_decay_factor',
                         'min_lr', 'experience_replay_capacity', 'batch_size']
        args = {}
        for key in args_dict.keys():
            if key in possible_args:
                args[key] = args_dict[key]
        return args

    def _check_arguments(self):
        if self.epsilon < 0 or self.epsilon > 1:
            raise ValueError('epsilon value should be between 0 to 1')
        if self.epsilon_decay_factor < 0 or self.epsilon_decay_factor > 1:
            raise ValueError('epsilon_decay_factor value should be between 0 to 1')
        if self.epsilon_decay_steps < 1 or not isinstance(self.epsilon_decay_steps, int):
            raise ValueError('epsilon_decay_steps value should be integer bigger then  1')
        is_valid_layer = lambda l: isinstance(l, int) and l > 0
        if not hasattr(self.args['layers'], '__iter__') or not all([is_valid_layer(l) for l in self.args['layers']]):
            raise ValueError('layers argument should be iterable with positive integer elements')

    def initialize_q(self):
        self.q_network1 = self._build_model()
        self.q_network2 = clone_model(self.q_network1)
        # self.q_network2.set_weights(self.q_network1.get_weights())
        self.training_model1 = self._build_training_model(1)
        self.training_model2 = self._build_training_model(2)
        optimizer1 = Adam(learning_rate=self.learning_rate, clipnorm=.005)
        self.training_model1.compile(optimizer1, loss='mse')
        optimizer2 = Adam(learning_rate=self.learning_rate, clipnorm=.005)
        self.training_model2.compile(optimizer2, loss='mse')


    def get_action_by_policy(self, state):
        if self.is_1:
            q_for_state = self.q_network1.predict(np.expand_dims(state, axis=0))
        else:
            q_for_state = self.q_network2.predict(np.expand_dims(state, axis=0))
        if not self.train_mode or np.random.uniform(0, 1) > self.epsilon:
            action_index = np.argmax(q_for_state)
        else:
            action_index = np.random.randint(len(self.actions))
        return self.actions[action_index], np.squeeze(q_for_state)[action_index]

    def update_q(self, **kwars):
        state, action, reward, new_state, d = kwars['state'], kwars['action'], kwars['reward'],\
                                                   kwars['new_state'], kwars['d']
        self.cache_to_experience_replay(state, action, reward, new_state, d)
        if len(self.experience_replay) == 0:
            return
        states_0, actions, rewards_0, rewards_1, states_2, ds = self.get_batch()
        ds = np.asarray(ds)
        actions = np.asarray([self.actions_indices[a] for a in actions], dtype=np.int32)
        rewards_0 = np.asarray(rewards_0, dtype=np.float64)
        rewards_1 = np.asarray(rewards_1, dtype=np.float64)
        states_2 = np.asarray(states_2)
        states_0 = np.asarray(states_0)
        done_indices = np.where(ds)
        undone_indices = np.where(ds == False)
        ys = np.zeros_like(actions, dtype=np.float64)
        ys[done_indices] += rewards_0[done_indices] + rewards_1[done_indices] * self.discount_factor
        q_next = self.get_action_by_max(states_2[undone_indices])
        ys[undone_indices] += rewards_0[undone_indices] + rewards_1[undone_indices] * self.discount_factor + \
                              self.discount_factor**2 * q_next
        ys = ys
        actions = np.concatenate([np.indices(actions.shape).T, np.expand_dims(actions, axis=0).T], axis=1)
        if self.is_1:
            loss = self.training_model1.train_on_batch(x=[states_0, actions], y=ys)
        else:
            loss = self.training_model2.train_on_batch(x=[states_0, actions], y=ys)
        self.episode_losses.append(loss)
        self.step += 1
        if self.environment.is_done():
            self.last_step_cache = {'state': None, 'action': None, 'reward': None, 'new_state': None, 'd': None}
            self.episode += 1
            if self.episode % self.target_update_episodes == 0 and not self.plato_phase:
                self.is_1 = not self.is_1
            if not self.plato_phase:
                if self.environment.step_num == 500:
                    self.perfect_counter += 1
                else:
                    self.perfect_counter = 0
                if self.perfect_counter == self.target_update_episodes:
                    self.plato_phase = True
                    self.epsilon = 0
                    print('plato phase')
                    self.learning_rate = self.min_lr
                    K.set_value(self.training_model1.optimizer.learning_rate, self.min_lr)
                    K.set_value(self.training_model1.optimizer.beta_1, 0.5)
                    K.set_value(self.training_model1.optimizer.beta_2, 0.5)
                    K.set_value(self.training_model2.optimizer.learning_rate, self.min_lr)
                    K.set_value(self.training_model2.optimizer.beta_1, 0.5)
                    K.set_value(self.training_model2.optimizer.beta_2, 0.5)
                elif self.learning_rate > self.min_lr:
                    self.learning_rate = self.lr_decay_factor * self.learning_rate
                    K.set_value(self.training_model1.optimizer.learning_rate, self.learning_rate)
                    K.set_value(self.training_model2.optimizer.learning_rate, self.learning_rate)
            self.last_episode_loss = np.mean(self.episode_losses)
            self.episode_losses = []
        if self.step % self.epsilon_decay_steps == 0:
            self._update_epsilon()

    def _update_epsilon(self):
        if self.plato_phase:
            return
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.epsilon_decay_factor
        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

    def get_all_actions(self):
        return self.actions

    def get_action_by_max(self, state):
        if self.is_1:
            q_for_state = self.q_network2.predict(state)
        else:
            q_for_state = self.q_network1.predict(state)
        return np.max(q_for_state, axis=-1)

    def _build_model(self):
        i = Input(shape=self.states_shape)
        layers = self.args['layers']
        d = i
        for layer in layers:
            d = Dense(layer, activation='relu')(d)
            d = BatchNormalization()(d)
        o = Dense(len(self.actions), activation='linear')(d)

        return Model(inputs=i, outputs=o)

    def _build_training_model(self, network_num):
        i = Input(shape=self.states_shape)
        a = Input(shape=(2,), dtype='int32')
        if network_num == 1:
            o = self.q_network1(i)
        else:
            o = self.q_network2(i)
        o = tf.gather_nd(o, a)
        return Model(inputs=[i, a], outputs=[o])


    def get_batch(self):
        batch_size = min(self.batch_size, len(self.experience_replay))
        indices = np.random.choice(len(self.experience_replay), batch_size)
        experience_replay_batch = [self.experience_replay[i] for i in indices]
        states_0, actions_0, rewards_0, rewards_1, states_2, ds = map(list, zip(*experience_replay_batch))
        return states_0, actions_0, rewards_0, rewards_1, states_2, ds


    def cache_to_experience_replay(self, state, action, reward, new_state, d):
        state_0, action_0, reward_0, d_0 = self.last_step_cache['state'], self.last_step_cache['action'],\
                                                    self.last_step_cache['reward'], self.last_step_cache['d']
        self.last_step_cache['state'] = state
        self.last_step_cache['action'] = action
        self.last_step_cache['reward'] = reward
        self.last_step_cache['new_state'] = new_state
        self.last_step_cache['d'] = d
        if state_0 is None or d_0:
            return
        self.experience_replay.append((state_0, action_0, reward_0, reward, new_state, d))

        if len(self.experience_replay) > self.experience_replay_capacity:
            self.experience_replay.pop(0)