from hw3.agent_interface import AgentInterface
import numpy as np
from scipy.special import softmax
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Softmax, Concatenate

from tensorflow.keras.models import clone_model, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K

class Agent(AgentInterface):
    def __init__(self, environment, args_dict):
        super().__init__(environment, 'actor_critic')
        self.environment = environment
        self.input_length = 6
        self.output_length = 9
        self.actor_lr = args_dict['actor_learning_rate']
        self.critic_lr = args_dict['critic_learning_rate']
        self.actor_forward = None
        self.actor_backward = None
        self.critic_forward = None
        self.critic_backward = None
        self._build_and_compile_actor()
        self._build_and_compile_critic()

    def get_actions(self, state):
        full_action_space = self.actor_forward.predict(np.expand_dims(np.asarray(state), axis=0))
        return np.squeeze(softmax(full_action_space[:self.environment.get_action_state_size()]))

    def get_value(self, state):
        return self.critic_forward.predict(np.expand_dims(np.asarray(state), axis=0))

    def update_weights(self, state, action_index, reward, next_state, discount_factor, I):
        I = np.expand_dims(np.asarray(I), axis=0)
        target = reward + discount_factor * self.get_value(next_state)
        td_error = target - self.get_value(state)

        action_index = np.expand_dims(np.asarray(action_index), axis=0)
        state = np.expand_dims(np.asarray(state), axis=0)
        actor_loss = self.actor_backward.train_on_batch(x=[state, action_index, I], y=td_error)
        critic_loss = self.critic_backward.train_on_batch(x=[state, I], y=td_error)
        return actor_loss, critic_loss

    def _build_and_compile_actor(self):
        i_f = Input(shape=(self.input_length,))
        x_f = Dense(12, activation='relu')(i_f)
        x_f = Dense(64, activation='relu')(x_f)
        o_f = Dense(self.output_length)(x_f)
        self.actor_forward = Model(inputs=i_f, outputs=o_f)
        i_b = Input(shape=(self.input_length,))
        o_b = self.actor_forward(i_b)
        I = Input(shape=(1,))
        action_idx = Input(shape=(1,))
        o_b = Concatenate()([o_b, action_idx, I])
        self.actor_backward = Model(inputs=[i_b, action_idx, I], outputs=o_b)
        self.actor_backward.compile(Adam(self.actor_lr), loss=actor_loss)


    def _build_and_compile_critic(self):
        i_f = Input(shape=(self.input_length,))
        x_f = Dense(256, activation='relu')(i_f)
        x_f = Dense(64, activation='relu')(x_f)
        o_f = Dense(1)(x_f)
        self.critic_forward = Model(inputs=i_f, outputs=o_f)

        i_b = Input(shape=(self.input_length,))
        o_b = self.critic_forward(i_b)
        I = Input(shape=(1,))
        o_b = Concatenate()([o_b, I])
        self.critic_backward = Model(inputs=[i_b, I], outputs=o_b)
        self.critic_backward.compile(Adam(self.critic_lr), loss=critic_loss)


def actor_loss(td_error, y_pred):
    y, action_index, I = y_pred[:, : -2], tf.cast(y_pred[:, -2], dtype=tf.int32), y_pred[:, -1]
    # padding = tf.constant([[1, 0]])
    # action_index_pad = tf.pad(action_index, padding)
    # action_prob = tf.gather_nd(y, tf.expand_dims(action_index_pad, axis=0))
    # neg_log_prob = -K.log(action_prob)
    neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=tf.one_hot(action_index, depth=9))
    return I * neg_log_prob * td_error


def critic_loss(td_error, y_pred):
    y, I = y_pred[:, 0], y_pred[:, 1]
    return -I * y * td_error





