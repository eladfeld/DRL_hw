import argparse
from importlib import import_module
agent_args_names = ['epsilon', 'epsilon_decay_factor', 'epsilon_decay_steps']
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib import pyplot as plt
def parse_args():
    parser = argparse.ArgumentParser(description='q-learning arguments')
    parser.add_argument('-e', dest='environment', type=str, required=True,
                        help='environment name from Environment directory')
    parser.add_argument('-a', dest='agent', type=str, required=True,
                        help='agent name from Agent directory')
    parser.add_argument('-v', dest='visualization', type=str,
                        help='visualization name')
    parser.add_argument('--epsilon', dest='epsilon', type=float,
                        help='optional, epsilon value for epsilon greedy policy, legal range [0, 1).')
    parser.add_argument('--epsilon_decay_factor', dest='epsilon_decay_factor', type=float,
                        help='optional, epsilon decay factor value for epsilon greedy policy, legal range [0, 1].')
    parser.add_argument('--epsilon_decay_steps', dest='epsilon_decay_steps', type=int,
                        help='optional, epsilon decay factor value for epsilon greedy policy, legal range [0, 1].')
    parser.add_argument('--episodes', dest='episodes', type=int, default=5000,
                        help='optional, max episodes for q_learning')
    parser.add_argument('--steps', dest='steps', type=int, default=100,
                        help='optional, max steps to run with q_learning per episode')
    parser.add_argument('--discount_factor', dest='discount_factor', type=float, default=0.95,
                            help = 'optional, discount factor for q learning algorithm')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.1,
                         help='optional,learning rate')
    return parser.parse_args()



def main():
    args = vars(parse_args())
    episodes_to_plot = [499, 1999, 4999]
    agents_args = {arg: args[arg] for arg in agent_args_names if args[arg] is not None}
    agent_class = getattr(import_module('hw1.Agents.' + args['agent']), 'Agent')
    environment_class = getattr(import_module('hw1.Environments.' + args['environment']), 'Environment')
    environment = environment_class()
    agent = agent_class(environment, agents_args)
    do_visualization = args['visualization'] is not None
    visualization = None
    if do_visualization:
        visualization_class = getattr(import_module('hw1.Visualizations.' + args['visualization']), 'Visualization')
        visualization = visualization_class(environment, agent, args)
    print('Running q_learning with Environment: %s, Agent: %s' % (environment, agent))
    agent.initialize_q()
    for episode in range(args['episodes']):
        steps = 0
        rewards = []
        environment.initialize_state()
        while not environment.is_done() and steps < args['steps']:
            state = environment.get_state()
            action, q = agent.get_action_by_policy(state)
            reward = environment.step(action)
            rewards.append(reward)
            steps += 1
            if environment.is_done():
                target = reward
            else:
                new_state = environment.get_state()
                _, q_next = agent.get_action_by_max(new_state)
                target = reward + args['discount_factor'] * q_next
            new_q = (1 - args['learning_rate']) * q + args['learning_rate'] * target
            agent.update_q(state, action, new_q)
        if do_visualization:
            visualization.update(steps=steps, rewards=rewards)
            if episode in episodes_to_plot:
                visualization.plot_heatmap(agent.q_lookup_table, episode)
    if do_visualization:
        visualization.show()
    print('done')


if __name__ == '__main__':
    main()