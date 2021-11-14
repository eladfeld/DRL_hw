import argparse
from importlib import import_module
agent_args_names = ['epsilon', 'epsilon_decay_factor', 'epsilon_decay_steps']
def parse_args():
    parser = argparse.ArgumentParser(description='q-learning arguments')
    parser.add_argument('-e', dest='environment', type=str, required=True,
                        help='environment name from Environment directory')
    parser.add_argument('-a', dest='agent', type=str, required=True,
                        help='agent name from Agent directory')
    parser.add_argument('--epsilon', dest='epsilon', type=float,
                        help='optional, epsilon value for epsilon greedy policy, legal range [0, 1).')
    parser.add_argument('--epsilon_decay_factor', dest='epsilon_decay_factor', type=float,
                        help='optional, epsilon decay factor value for epsilon greedy policy, legal range [0, 1].')
    parser.add_argument('--epsilon_decay_steps', dest='epsilon_decay_steps', type=int,
                        help='optional, steps amount until epsilon decay factor value for epsilon greedy policy,'
                             ' integer bigger then 0.')
    return parser.parse_args()
def main():
    args = vars(parse_args())
    agents_args = {arg: args[arg] for arg in agent_args_names if args[arg] is not None}


    agent_class = getattr(import_module('Agents.' + args['agent']), 'Agent')
    environment_class = getattr(import_module('Environments.' + args['environment']), 'Environment')
    environment = environment_class()
    agent = agent_class(environment, agents_args)

    print('Running q_learning with Environment: %s, Agent: %s' % (environment, agent))
    print('done')
if __name__ == '__main__':
    main()