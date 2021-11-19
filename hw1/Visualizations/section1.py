from hw1.visualization_interface import VisualizationInterface
from matplotlib import pyplot as plt
import numpy as np
class Visualization(VisualizationInterface):
    def __init__(self, environment, agent, args_dict):
        super().__init__(environment, agent, args_dict, name='section1')
        self.environment = environment
        self.agent = agent
        self.episode = 0
        self.max_steps = args_dict['steps']
        self.q_cache = []
        self.steps_to_cache_q = [500, 2000, args_dict['episodes']]
        self.rewards = []
        self.steps = []
        self.final_state = len(self.environment.get_all_states()) -1

    def update(self, **kwargs):
        self.episode += 1
        steps = kwargs['steps']
        reward = kwargs['rewards'][-1]
        self.rewards.append(reward)
        steps = steps if self.environment.get_state() == self.final_state else self.max_steps
        self.steps.append(steps)
        if self.episode in self.steps_to_cache_q:
            self.q_cache.append(self.agent.q_lookup_table)

    def show(self):
        avg_steps = 100
        steps_avgs = [np.mean(self.steps[i* avg_steps: (i+1) * avg_steps]) for i in range(self.episode // avg_steps)]
        rewards_avgs = [np.mean(self.rewards[i* avg_steps: (i+1) * avg_steps]) for i in range(self.episode // avg_steps)]

        fig, ax = plt.subplots(2)
        fig.set_size_inches(16, 8)
        ax[0].plot(np.linspace(1, self.episode +1, self.episode), self.rewards)
        ax[0].plot(np.linspace(avg_steps, (len(rewards_avgs) + 1) * 100, len(rewards_avgs)), rewards_avgs, color='red',
                   linewidth=4)
        ax[0].set_ylabel('reward')
        ax[0].set_title('reward for episode')
        ax[0].legend(['reward', 'average reward over %s episodes' % avg_steps])

        ax[1].plot(np.linspace(avg_steps, (len(steps_avgs) + 1) * 100, len(steps_avgs)), steps_avgs,
                   linewidth=4)
        ax[1].set_ylabel('reward')
        ax[1].set_xlabel('episode')
        ax[1].set_title('avg steps for episode')
        ax[1].legend(['average steps over %s episodes' % avg_steps])
        plt.show()