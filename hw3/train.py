import argparse
from importlib import import_module
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def parse_args():
    parser = argparse.ArgumentParser(description='q-learning arguments')
    parser.add_argument('--e', dest='environment', type=str, required=True,
                        help='environment name from Environment directory')
    parser.add_argument('--a', dest='agent', type=str, required=True,
                        help='agent name from Agent directory')
    parser.add_argument('--experiment', dest='experiment', type=str,
                        help='experiment name')
    parser.add_argument('--episodes', dest='episodes', type=int, default=5000,
                        help='optional, max episodes for q_learning')
    parser.add_argument('--steps', dest='steps', type=int, default=200,
                        help='optional, max steps to run with q_learning per episode')
    parser.add_argument('--discount_factor', dest='discount_factor', type=float, default=0.95,
                            help = 'optional, discount factor for q learning algorithm')
    parser.add_argument('--a-lr', dest='actor_learning_rate', type=float, default=0.0002,
                         help='actor learning rate')
    parser.add_argument('--c-lr', dest='critic_learning_rate', type=float, default=0.002,
                         help='actor learning rate')


    return parser.parse_args()



def main():
    args = vars(parse_args())
    agent_class = getattr(import_module('hw3.Agents.' + args['agent']), 'Agent')
    environment_class = getattr(import_module('hw3.Environments.' + args['environment']), 'Environment')
    environment = environment_class(args)
    agent = agent_class(environment, args)
    run_experiment = args['experiment'] is not None
    for episode in range(args['episodes']):
        I = 1
        environment.initialize_state()
        rewards = []
        environment.render()
        for step in range(args['steps']):
            state = environment.get_state()
            actions = agent.get_actions(state)
            action = np.random.choice(environment.get_action_state_size(), p=actions)
            reward = environment.step(action)
            next_state = environment.get_state()
            environment.render()
            if environment.is_done():
                break
            reward_for_train = np.abs(state[1])*10 + np.abs(state[0] + 0.45) + reward
            actor_loss, critic_loss = agent.update_weights(state, action, reward_for_train, next_state, args['discount_factor'], I)
            rewards.append(reward_for_train)

            I *= args['discount_factor']
        print(np.sum(rewards))

if __name__ == '__main__':
    main()