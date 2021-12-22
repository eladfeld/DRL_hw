from hw1.experiment_interface import ExperimentInterface
from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf


class Experiment(ExperimentInterface):
    def __init__(self, environment, agent, args_dict):
        super().__init__(environment, agent, args_dict, name='section2')
        self.environment = environment
        self.agent = agent
        self.episode = 0
        self.max_steps = args_dict['steps']
        self.rewards = []
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'section2')
        self.writer = tf.summary.create_file_writer(log_path)
        print('saving logs to: %s' % log_path)

    def update(self, **kwargs):
        self.episode += 1
        reward = np.sum(kwargs['rewards'])
        self.rewards.append(reward)
        loss = self.agent.last_episode_loss
        print('episode: %d, loss: %1.2f, reward %d' % (self.episode, loss, reward))
        with self.writer.as_default():
            tf.summary.scalar("loss", loss, step=self.episode)
            tf.summary.scalar("reward", reward, step=self.episode)
            self.writer.flush()


    def show(self):
        avg_steps = 20
        rewards_avgs = [np.mean(self.rewards[i * avg_steps: (i + 1) * avg_steps]) for i in
                        range(self.episode // avg_steps)]

        fig = plt.figure()
        fig.set_size_inches(16, 8)
        plt.plot(np.linspace(1, self.episode + 1, self.episode), self.rewards)
        plt.plot(np.linspace(avg_steps, (len(rewards_avgs) + 1) * avg_steps, len(rewards_avgs)), rewards_avgs, color='red',
                   linewidth=4)
        plt.ylabel('reward')
        plt.title('reward for episode')
        plt.legend(['reward', 'average reward over %s episodes' % avg_steps])
        plt.show()


    def is_done(self):
        return len(self.rewards) > 100 and np.mean(self.rewards[-101: -1]) >= 475

    def _write_log(self, names, logs, episode_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.critic.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.tensorboard.writer.add_summary(summary, episode_no)
            self.tensorboard.writer.flush()