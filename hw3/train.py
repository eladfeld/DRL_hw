import argparse
from importlib import import_module
import sys
import os
import numpy as np
import tensorflow as tf
from datetime import date
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def parse_args():
    parser = argparse.ArgumentParser(description='q-learning arguments')
    parser.add_argument('--e', dest='environment', type=str, required=True,
                        help='environment name from Environment directory')
    parser.add_argument('--a', dest='agent', type=str, required=True,
                        help='agent name from Agent directory')
    parser.add_argument('--episodes', dest='episodes', type=int, default=2000,
                        help='optional, max episodes ')
    parser.add_argument('--steps', dest='steps', type=int, default=1000,
                        help='optional, max steps  per episode')
    parser.add_argument('--discount_factor', dest='discount_factor', type=float, default=0.99,
                            help = 'optional, discount factor')
    parser.add_argument('--a-lr', dest='actor_learning_rate', type=float, default=4e-4,
                         help='actor learning rate')
    parser.add_argument('--c-lr', dest='critic_learning_rate', type=float, default=2e-3,
                         help='actor learning rate')
    parser.add_argument('--transfer', dest='do_transfer', action='store_true', default=False,
                         help='use transfer learning')
    parser.add_argument('--w', dest='initial_weights', type=str, nargs='+', default='',
                        help='path to initial weights')
    parser.add_argument('--render', dest='render', action='store_true', default=False,
                         help='render environment')


    return parser.parse_args()



def main():
    args = vars(parse_args())
    agent_class = getattr(import_module('hw3.Agents.' + args['agent']), 'Agent')
    environment_class = getattr(import_module('hw3.Environments.' + args['environment']), 'Environment')
    environment = environment_class(args)
    agent = agent_class(environment, args)
    today = date.today().strftime("%d_%m_%Y")
    trans_string = '_transfer' if args['do_transfer'] else ''
    out_path = os.path.join(os.path.dirname(__file__), 'models',
                            'env_' + args['environment'] + '_agent_' + args['agent'] + trans_string, today)
    log_path = os.path.join(out_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    writer = tf.summary.create_file_writer(log_path)
    start_time = time.time()
    for episode in range(args['episodes']):
        I = 1
        environment.initialize_state()
        external_rewards = []
        intrinsic_rewards = []
        actor_losses = []
        critic_losses = []
        for step in range(args['steps']):
            state = environment.get_state()
            action = agent.get_actions(state)
            external_reward, intrinsic_reward = environment.step(action)
            next_state = environment.get_state()
            if args['render']:
                environment.render()
            actor_loss, critic_loss = agent.update_weights(state, action, external_reward + intrinsic_reward,
                                                           next_state, args['discount_factor'], I)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            external_rewards.append(external_reward)
            intrinsic_rewards.append(intrinsic_reward)
            I *= args['discount_factor']
            if environment.is_done():
                break
        ep_intrinsic_reward = np.sum(intrinsic_rewards)
        ep_external_reward = np.sum(external_rewards)
        ep_actor_loss = np.mean(actor_losses)
        ep_critic_loss = np.mean(critic_losses)
        use_intrinsic_reward = environment.use_intrinsic_rewards()
        with writer.as_default():
            if use_intrinsic_reward:
                tf.summary.scalar("external_reward", ep_external_reward, step=episode)
                tf.summary.scalar("intrinsic_reward", ep_intrinsic_reward, step=episode)
            else:
                tf.summary.scalar("reward", ep_external_reward, step=episode)
            tf.summary.scalar("actor_loss", ep_critic_loss, step=episode)
            tf.summary.scalar("critic_loss", ep_critic_loss, step=episode)
            writer.flush()
        if not use_intrinsic_reward:
            print('episode: %d, reward, %1.2f, actor_loss:%1.5f, critic_loss: %1.5f' %
                  (episode, ep_external_reward, ep_actor_loss, ep_critic_loss))
        else:
            print('episode: %d, external_reward, %1.2f, intrinsic_reward, %1.2f, actor_loss:%1.5f, critic_loss: %1.5f'
                  % (episode, ep_external_reward, ep_intrinsic_reward, ep_actor_loss, ep_critic_loss))
        if environment.is_converge():
            break
        agent.save_weights(out_path)
    run_time = time.time() - start_time
    print('run time: %d hours, %d minutes, % 1.2f seconds' % (int(run_time//3600), int(run_time//60) % 60,
                                                             run_time - int(run_time//60) * 60))
    agent.save_weights(out_path)
    print('weights saved in %s' % out_path)
if __name__ == '__main__':
    main()