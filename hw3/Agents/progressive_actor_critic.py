from hw3.agent_interface import AgentInterface
import numpy as np
from scipy.special import softmax
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Lambda
import os
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from hw3.nn import  get_progressive_actor, get_progressive_critic
from tensorflow.keras import backend as K

class Agent(AgentInterface):
    def __init__(self, environment, args_dict):
        super().__init__(environment, 'actor_critic')
        self.environment = environment
        self.action_state_size = self.environment.get_action_state_size()
        self.actor_lr = args_dict['actor_learning_rate']
        self.critic_lr = args_dict['critic_learning_rate']
        self.initial_weights_paths = args_dict['initial_weights']
        self.actor_forward = None
        self.actor_backward = None
        self.critic_forward = None
        self.critic_backward = None
        self.input_shape = 6
        self.output_shape = 3
        self._build_and_compile_actor()
        self._build_and_compile_critic()
        self.iteration = 0

    def get_actions(self, state):
        full_action_space = np.squeeze(self.actor_forward.predict(np.expand_dims(np.asarray(state), axis=0)))
        actions_prob = softmax(full_action_space[:self.action_state_size])
        action = np.random.choice(self.action_state_size, p=actions_prob)
        return action

    def get_value(self, state):
        return self.critic_forward.predict(np.expand_dims(np.asarray(state), axis=0))

    def update_weights(self, state, action_index, reward, next_state, discount_factor, I):
        self.iteration += 1
        I = np.expand_dims(np.asarray(I), axis=0)
        next_value = 0 if self.environment.is_done() else self.get_value(next_state)
        target = reward + discount_factor * next_value
        td_error = target - self.get_value(state)

        action = np.expand_dims(np.asarray(np.eye(self.action_state_size)[action_index]), axis=0)
        state = np.expand_dims(np.asarray(state), axis=0)
        actor_loss = self.actor_backward.train_on_batch(x=[state, action, I], y=td_error)
        critic_loss = self.critic_backward.train_on_batch(x=[state, I], y=td_error)

        return actor_loss, critic_loss

    def _build_and_compile_actor(self):
        self.actor_forward = get_progressive_actor(*self.initial_weights_paths)
        i_b = Input(shape=self.actor_forward.input_shape[1:])
        o_b = self.actor_forward(i_b)
        o_b = Lambda(lambda x: x[:, :self.action_state_size])(o_b)
        I = Input(shape=(1,))
        action = Input(shape=(self.action_state_size,))
        o_b = Concatenate()([o_b, action, I])
        self.actor_backward = Model(inputs=[i_b, action, I], outputs=o_b)
        self.actor_backward.compile(Adam(self.actor_lr), loss=get_actor_loss(self.action_state_size))


    def _build_and_compile_critic(self):

        self.critic_forward = get_progressive_critic(*self.initial_weights_paths)
        i_b = Input(shape=self.critic_forward.input_shape[1:])
        o_b = self.critic_forward(i_b)
        I = Input(shape=(1,))
        o_b = Concatenate()([o_b, I])
        self.critic_backward = Model(inputs=[i_b, I], outputs=o_b)
        self.critic_backward.compile(Adam(self.critic_lr), loss=critic_loss)

    def save_weights(self, path):
        self.actor_forward.save_weights(os.path.join(path, 'actor.h5'))
        self.critic_forward.save_weights(os.path.join(path, 'critic.h5'))

    def load_and_freeze_actor(self):
        self.actor_forward.load_weights(os.path.join(self.initial_weights_path, 'actor.h5'))
        for layer in self.actor_forward.layers[:-1]:
            layer.trainable = False

    def load_and_freeze_critic(self):
        self.critic_forward.load_weights(os.path.join(self.initial_weights_path, 'critic.h5'))
        for layer in self.critic_forward.layers[:-1]:
            layer.trainable = False




def get_actor_loss(depth):
    def actor_loss(td_error, y_pred):
        y, action, I = y_pred[:, : depth], y_pred[:, depth: -1], y_pred[:, -1]
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=action)
        return I * neg_log_prob * td_error
    return actor_loss


def critic_loss(td_error, y_pred):
    y, I = y_pred[:, 0], y_pred[:, 1]
    return -I * y * td_error





