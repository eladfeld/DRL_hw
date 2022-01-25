from hw3.agent_interface import AgentInterface
import numpy as np
from scipy.special import softmax
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from hw3.nn import get_actor, get_critic
from tensorflow.keras import backend as K

class Agent(AgentInterface):
    def __init__(self, environment, args_dict):
        super().__init__(environment, 'actor_critic')
        self.environment = environment
        self.action_state_size = self.environment.get_action_state_size()
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
        actions_prob = np.squeeze(softmax(full_action_space[:self.action_state_size]))
        action = np.random.choice(self.action_state_size, p=actions_prob)
        return action

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
        self.actor_forward = get_actor()
        i_b = Input(shape=self.actor_forward.input_shape[1:])
        o_b = self.actor_forward(i_b)
        I = Input(shape=(1,))
        action_idx = Input(shape=(1,))
        o_b = Concatenate()([o_b, action_idx, I])
        self.actor_backward = Model(inputs=[i_b, action_idx, I], outputs=o_b)
        self.actor_backward.compile(Adam(self.actor_lr), loss=get_actor_loss(self.action_state_size))


    def _build_and_compile_critic(self):

        self.critic_forward = get_critic()

        i_b = Input(shape=self.critic_forward.input_shape[1:])
        o_b = self.critic_forward(i_b)
        I = Input(shape=(1,))
        o_b = Concatenate()([o_b, I])
        self.critic_backward = Model(inputs=[i_b, I], outputs=o_b)
        self.critic_backward.compile(Adam(self.critic_lr), loss=critic_loss)

    def save_weights(self,path):
        self.actor_forward.save_wheights(path+'_actor.h5')
        self.critic_forward.save_wheights(path + '_critic.h5')

def get_actor_loss(depth):
    def actor_loss(td_error, y_pred):
        y, action_index, I = y_pred[:, : -2], tf.cast(y_pred[:, -2], dtype=tf.int32), y_pred[:, -1]
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=tf.one_hot(action_index, depth=depth))
        return I * neg_log_prob * td_error
    return actor_loss


def critic_loss(td_error, y_pred):
    y, I = y_pred[:, 0], y_pred[:, 1]
    return -I * y * td_error





