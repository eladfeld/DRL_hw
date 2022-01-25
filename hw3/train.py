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
    parser.add_argument('--steps', dest='steps', type=int, default=5000,
                        help='optional, max steps to run with q_learning per episode')
    parser.add_argument('--discount_factor', dest='discount_factor', type=float, default=0.95,
                            help = 'optional, discount factor for q learning algorithm')
    parser.add_argument('--a-lr', dest='actor_learning_rate', type=float, default=0.0001,
                         help='actor learning rate')
    parser.add_argument('--c-lr', dest='critic_learning_rate', type=float, default=0.001,
                         help='actor learning rate')
    parser.add_argument('--transfer', dest='do_transfer', type=bool, action='store_true', default=False,
                         help='use transfer learning')
    parser.add_argument('--w', dest='initial_weights', type=str, default='',
                        help='path to initial weights')


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
        external_rewards = []
        intrinsic_rewards = []
        actor_losses = []
        critic_losses = []
        environment.render()
        for step in range(args['steps']):
            state = environment.get_state()
            action = agent.get_actions(state)
            external_reward, intrinsic_reward = environment.step(action)
            next_state = environment.get_state()
            environment.render()
            if environment.is_done():
                break
            actor_loss, critic_loss = agent.update_weights(state, action, external_reward + intrinsic_reward,
                                                           next_state, args['discount_factor'], I)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            external_rewards.append(external_reward)
            intrinsic_rewards.append(intrinsic_reward)
            I *= args['discount_factor']
        ep_intrinsic_return = np.sum(intrinsic_rewards)
        ep_external_return = np.sum(external_rewards)
        if ep_intrinsic_return == 0:
            print('episode: %d, reward, %1.2f, actor_loss:%1.5f, critic_loss: %1.5f' % (episode,
                  ep_external_return, np.mean(actor_losses), np.mean(critic_losses)))
        else:
            print('episode: %d, external_reward, %1.2f, intrinsic_reward, %1.2f, actor_loss:%1.5f, critic_loss: %1.5f'
                  % (episode, ep_external_return, ep_intrinsic_return, np.mean(actor_losses),
                     np.mean(critic_losses)))

    agent.save_weights(os.path.join(os.path.basename(__file__), 'env_' + args['environment'] + '_agent_' + args['agent']))

if __name__ == '__main__':
    main()