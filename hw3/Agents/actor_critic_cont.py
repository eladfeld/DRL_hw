from hw3.agent_interface import AgentInterface
import numpy as np
from scipy.special import softmax
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Lambda, Softmax
from tensorflow.keras.initializers import GlorotNormal
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
        self.actor_optimizer = Adam(self.actor_lr)
        self.critic_optimizer = Adam(self.critic_lr)

    def get_actions(self, state):
        full_action_space = np.squeeze(self.actor_forward.predict(np.expand_dims(np.asarray(state), axis=0)))\

        mu = full_action_space[0]
        sigma = np.log(np.exp(full_action_space[1]) + 1)
        action = np.random.normal(mu, sigma)
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
        actor_loss = self._train_actor(I, action, state, td_error)
        critic_loss = self._train_critic(I, state, td_error)
        if self.environment.is_done():
            self.actor_lr *= 0.95
            K.set_value(self.actor_optimizer.learning_rate, self.actor_lr)

        return actor_loss, critic_loss

    def _train_actor(self, I, action, state, td_error):
        with tf.GradientTape() as tape:
            logits = self.actor_backward([state, action, I], training=True)
            loss_value = self.actor_loss(td_error, logits)
            grads = tape.gradient(loss_value, self.actor_backward.trainable_weights)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor_backward.trainable_weights))
        return float(loss_value)


    def _train_critic(self, I, state, td_error):
        with tf.GradientTape() as tape:
            logits = self.critic_backward([state, I], training=True)
            loss_value = self.critic_loss(td_error, logits)
            grads = tape.gradient(loss_value, self.critic_backward.trainable_weights)
            self.actor_optimizer.apply_gradients(zip(grads, self.critic_backward.trainable_weights))
        return float(loss_value)

    def _build_and_compile_actor(self):
        self.actor_forward = get_actor()
        if self.is_transfer:
            self.load_and_freeze_actor()
        i_b = Input(shape=self.actor_forward.input_shape[1:])
        o_b = self.actor_forward(i_b)
        I = Input(shape=(1,))
        action = Input(shape=(1,))
        o_b = Concatenate()([o_b, action, I])
        self.actor_backward = Model(inputs=[i_b, action, I], outputs=o_b)


    def _build_and_compile_critic(self):

        self.critic_forward = get_critic()
        if self.is_transfer:
            self.load_and_freeze_critic()
        i_b = Input(shape=self.critic_forward.input_shape[1:])
        o_b = self.critic_forward(i_b)
        I = Input(shape=(1,))
        o_b = Concatenate()([o_b, I])
        self.critic_backward = Model(inputs=[i_b, I], outputs=o_b)

    def save_weights(self, path):
        self.actor_forward.save_weights(os.path.join(path, 'actor.h5'))
        self.critic_forward.save_weights(os.path.join(path, 'critic.h5'))

    def load_and_freeze_actor(self):
        self.actor_forward.load_weights(os.path.join(self.initial_weights_path[0], 'actor.h5'))
        for i in range(len(self.actor_forward.layers[:-1])):
            self.actor_forward.layers[i].trainable = False
        self.actor_forward.layers[-1].set_weights([GlorotNormal()(shape=w.shape) for w in
                                                    self.actor_forward.layers[-1].get_weights()])

    def load_and_freeze_critic(self):
        self.critic_forward.load_weights(os.path.join(self.initial_weights_path[0], 'critic.h5'))
        for i in range(len(self.critic_forward.layers[:-1])):
            self.critic_forward.layers[i].trainable = False
        self.critic_forward.layers[-1].set_weights([GlorotNormal()(shape=w.shape) for w in
                                                    self.critic_forward.layers[-1].get_weights()])

    def actor_loss(self, td_error, y_pred):
        mu, sigma, action, I = y_pred[:, 0], y_pred[:, 1], y_pred[:, -2], y_pred[:, -1]
        dist = tfp.distributions.Normal(mu, tf.nn.softplus(sigma))
        neg_log_prob = -dist.log_prob(action)
        return I * neg_log_prob * td_error

    def critic_loss(self, td_error, y_pred):
        y, I = y_pred[:, 0], y_pred[:, 1]
        return -I * y * td_error







