from hw3.agent_interface import AgentInterface
import numpy as np
from scipy.special import softmax
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Lambda, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from hw3.nn import get_actor, get_critic
import os
import tensorflow.keras.backend as K
class Agent(AgentInterface):
    def __init__(self, environment, args_dict):
        super().__init__(environment, 'actor_critic_continuous')
        self.environment = environment
        self.actor_lr = args_dict['actor_learning_rate']
        self.critic_lr = args_dict['critic_learning_rate']
        self.is_transfer = args_dict['do_transfer']
        self.initial_weights_path = args_dict['initial_weights']
        self.actor_forward = None
        self.actor_backward = None
        self.critic_forward = None
        self.critic_backward = None
        self._build_and_compile_actor()
        self._build_and_compile_critic()

    def get_actions(self, state):
        epsilon = 1e-4
        full_action_space = softmax(np.squeeze(self.actor_forward.predict(np.expand_dims(np.asarray(state), axis=0)))[:-1] * 3)\
                            * 10
        alpha = full_action_space[0] + epsilon
        beta = full_action_space[1] + epsilon
        action = np.random.beta(alpha, beta)
        return action

    def get_value(self, state):
        return self.critic_forward.predict(np.expand_dims(np.asarray(state), axis=0))

    def update_weights(self, state, action, reward, next_state, discount_factor, I):
        I = np.expand_dims(np.asarray(I), axis=0)
        next_value = 0 if self.environment.is_done() else self.get_value(next_state)
        target = reward + discount_factor * next_value
        td_error = target - self.get_value(state)
        action = np.expand_dims(np.asarray(action), axis=0)
        state = np.expand_dims(np.asarray(state), axis=0)
        actor_loss = self.actor_backward.train_on_batch(x=[state, action, I], y=td_error)
        critic_loss = self.critic_backward.train_on_batch(x=[state, I], y=td_error)
        if self.environment.is_done() and self.environment.is_decay():
            self.actor_lr *= 0.1
            self.critic_lr *= 0.1
            K.set_value(self.actor_backward.optimizer.learning_rate, self.actor_lr)
            K.set_value(self.critic_backward.optimizer.learning_rate, self.critic_lr)
        return actor_loss, critic_loss

    def _build_and_compile_actor(self):
        self.actor_forward = get_actor()
        if self.is_transfer:
            self.load_and_freeze_actor()
        i_b = Input(shape=self.actor_forward.input_shape[1:])
        o_b = self.actor_forward(i_b)
        o_b = Lambda(lambda x: x[:, :-1] * 3)(o_b)
        o_b = Softmax()(o_b)
        o_b = Lambda(lambda x: x * 10)(o_b)
        I = Input(shape=(1,))
        action = Input(shape=(1,))
        o_b = Concatenate()([o_b, action, I])
        self.actor_backward = Model(inputs=[i_b, action, I], outputs=o_b)
        self.actor_backward.compile(Adam(self.actor_lr, clipnorm=.1), loss=actor_loss)


    def _build_and_compile_critic(self):

        self.critic_forward = get_critic()
        if self.is_transfer:
            self.load_and_freeze_critic()
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


def actor_loss(td_error, y_pred):
    def get_log_prob(alpha, beta, x):
        return tf.math.lgamma(alpha + beta) - tf.math.lgamma(alpha) - tf.math.lgamma(beta) + \
                (alpha - 1) * tf.math.log(x) + (beta - 1) * tf.math.log(1 - x)


    alpha, beta, action, I = y_pred[:, 0], y_pred[:, 1], y_pred[:, -2], y_pred[:, -1]
    neg_log_prob = get_log_prob(alpha, beta, alpha / (alpha + beta)) - get_log_prob(alpha, beta, action)
    return I * neg_log_prob * td_error


def critic_loss(td_error, y_pred):
    y, I = y_pred[:, 0], y_pred[:, 1]
    return -I * y * td_error





