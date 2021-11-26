from hw1.experiment_interface import ExperimentInterface
from matplotlib import pyplot as plt
import numpy as np


class Experiment(ExperimentInterface):
    def __init__(self, environment, agent, args_dict):
        super().__init__(environment, agent, args_dict, name='section2')
        self.environment = environment
        self.agent = agent
        self.episode = 0
        self.max_steps = args_dict['steps']
        self.rewards = []

    def update(self, **kwargs):
        self.episode += 1
        self.rewards.append(len(kwargs['rewards']))


    def show(self):
        avg_steps = 100
        rewards_avgs = [np.mean(self.rewards[i * avg_steps: (i + 1) * avg_steps]) for i in
                        range(self.episode // avg_steps)]

        fig = plt.figure()
        fig.set_size_inches(16, 8)
        plt.plot(np.linspace(1, self.episode + 1, self.episode), self.rewards)
        plt.plot(np.linspace(avg_steps, (len(rewards_avgs) + 1) * 100, len(rewards_avgs)), rewards_avgs, color='red',
                   linewidth=4)
        plt.ylabel('reward')
        plt.title('reward for episode')
        plt.legend(['reward', 'average reward over %s episodes' % avg_steps])
        plt.show()


    def is_done(self):
        return len(self.rewards) > 100 and np.mean(self.rewards[-101: -1]) >= 475