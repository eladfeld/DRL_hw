import random
random.seed(1)
import numpy as np
np.random.seed(1)
import tensorflow.compat.v1 as tf
tf.random.set_random_seed(1)
import gym
import os
tf.disable_v2_behavior()


env = gym.make('CartPole-v1')

class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.td_error = tf.placeholder(tf.float32, name="td_error")
            self.I = tf.placeholder(tf.float32, name="I")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.I * self.neg_log_prob * self.td_error)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

class ValueNetwork:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.td_error = tf.placeholder(tf.float32, name="target")
            self.I = tf.placeholder(tf.float32, name="I")

            self.W1 = tf.get_variable("W1", [self.state_size, 256], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [256], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [256, 64], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [64], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3", [64, 1], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b3 = tf.get_variable("b3", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            self.loss = tf.reduce_mean(-self.I * self.output * self.td_error)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

# Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
critic_learning_rate = 0.002
actor_learning_rate = 0.0004
render = False

# Initialize the actor network
tf.reset_default_graph()
actor = PolicyNetwork(state_size, action_size, actor_learning_rate)
critic = ValueNetwork(state_size, critic_learning_rate)

# tensorboard logs
actor_loss_placeholder = tf.compat.v1.placeholder(tf.float32)
tf.compat.v1.summary.scalar(name="policy_losses", tensor=actor_loss_placeholder)
critic_loss_placeholder = tf.compat.v1.placeholder(tf.float32)
tf.compat.v1.summary.scalar(name="value_losses", tensor=actor_loss_placeholder)
reward_placeholder = tf.compat.v1.placeholder(tf.float32)
tf.compat.v1.summary.scalar(name="reward", tensor=reward_placeholder)
avg_reward_placeholder = tf.compat.v1.placeholder(tf.float32)
tf.compat.v1.summary.scalar(name="avg_reward", tensor=avg_reward_placeholder)
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
if not os.path.isdir(log_path):
    os.mkdir(log_path)
writer = tf.compat.v1.summary.FileWriter(log_path)
summaries = tf.compat.v1.summary.merge_all()
print('saving logs to: %s' % log_path)

# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    solved = False
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0
    episode_critic_loss = []
    episode_actor_loss = []
    for episode in range(max_episodes):
        state = env.reset()
        # state = np.concatenate([state, np.asarray([0])])
        state = state.reshape([1, state_size])
        episode_transitions = []
        I=1
        for step in range(max_steps):
            value = sess.run(critic.output, {critic.state: state})
            actions_distribution = sess.run(actor.actions_distribution, {actor.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step(action)
            # next_state = np.concatenate([next_state, np.asarray([(step + 1) / max_steps])])
            next_state = next_state.reshape([1, state_size])
            next_value = sess.run(critic.output, {critic.state: next_state}) if not done else 0

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1

            episode_rewards[episode] += reward

            target = reward + discount_factor * next_value
            td_error = target - value

            value_feed_dict = {critic.state: state, critic.td_error: td_error, critic.I: I}
            _, critic_loss = sess.run([critic.optimizer, critic.loss], value_feed_dict)
            policy_feed_dict = {actor.state: state, actor.td_error: td_error, actor.action: action_one_hot,actor.I: I}
            _, actor_loss = sess.run([actor.optimizer, actor.loss], policy_feed_dict)
            state = next_state
            if done:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break
            I = I * discount_factor

        if solved:
            break


        avg_actor_loss = np.mean(episode_actor_loss)
        avg_critic_loss = np.mean(episode_critic_loss)
        summery = sess.run(summaries, feed_dict={actor_loss_placeholder: avg_actor_loss,
                                                 critic_loss_placeholder: avg_critic_loss,
                                                 reward_placeholder: episode_rewards[episode],
                                                 avg_reward_placeholder: average_rewards if episode > 98 else 0})
        writer.add_summary(summery, global_step=episode)
