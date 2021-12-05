import argparse
from importlib import import_module
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
deep_agents = ['dqn_cart', 'dueling_dqn_cart']
np.random.seed(0)
def parse_args():
    parser = argparse.ArgumentParser(description='q-learning arguments')
    parser.add_argument('-e', dest='environment', type=str, required=True,
                        help='environment name from Environment directory')
    parser.add_argument('-a', dest='agent', type=str, required=True,
                        help='agent name from Agent directory')
    parser.add_argument('--experiment', dest='experiment', type=str,
                        help='experiment name')
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0,
                        help='optional, epsilon value for epsilon greedy policy, legal range [0, 1).')
    parser.add_argument('--epsilon_decay_factor', dest='epsilon_decay_factor', type=float, default=1.,
                        help='optional, epsilon decay factor value for epsilon greedy policy, legal range [0, 1].')
    parser.add_argument('--epsilon_decay_steps', dest='epsilon_decay_steps', type=int, default=1,
                        help='optional, epsilon decay factor value for epsilon greedy policy, legal range [0, 1].')
    parser.add_argument('--min_epsilon', dest='min_epsilon', type=float, default=5e-3,
                        help='min epsilon value')
    parser.add_argument('--episodes', dest='episodes', type=int, default=5000,
                        help='optional, max episodes for q_learning')
    parser.add_argument('--steps', dest='steps', type=int, default=100,
                        help='optional, max steps to run with q_learning per episode')
    parser.add_argument('--discount_factor', dest='discount_factor', type=float, default=0.95,
                            help = 'optional, discount factor for q learning algorithm')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.1,
                         help='optional,learning rate')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=512,
                        help='optional, batch size for dqn training')
    parser.add_argument('--layers', dest='layers', type=int, nargs='+', default=[32, 32, 32],
                        help='optional, hidden layers for dqn network')
    parser.add_argument('--target_update_episodes', dest='target_update_episodes', type=int, default=2,
                        help='optional, steps to update dqn target network')
    parser.add_argument('--experience_replay_capacity', dest='experience_replay_capacity', type=int, default=50000,
                        help='optional, steps to update dqn target network')
    parser.add_argument('--lr_decay_factor', dest='lr_decay_factor', type=float, default=1.,
                        help='optional, decay factor for learning rate decay')
    parser.add_argument('--min_lr', dest='min_lr', type=float, default=1e-8,
                        help='min learning rate for dqn')
    return parser.parse_args()

def main():
    args = vars(parse_args())
    is_deep = args['agent'] in deep_agents
    if not is_deep:
        q_learning(args)
    else:
        deep_q_learning(args)

def deep_q_learning(args):
    agent_class = getattr(import_module('hw1.Agents.' + args['agent']), 'Agent')
    environment_class = getattr(import_module('hw1.Environments.' + args['environment']), 'Environment')
    environment = environment_class(args)
    agent = agent_class(environment, args)
    run_experiment = args['experiment'] is not None
    experiment = None
    if run_experiment:
        experiment_class = getattr(import_module('hw1.Experiments.' + args['experiment']), 'Experiment')
        experiment = experiment_class(environment, agent, args)
    print('Running q_learning with Environment: %s, Agent: %s' % (environment, agent))
    agent.initialize_q()
    experience_replay = []
    for episode in range(args['episodes']):
        episode_rewards = []
        steps = 0
        environment.initialize_state()
        while not environment.is_done() and steps < args['steps']:
            state = environment.get_state()
            action, _ = agent.get_action_by_policy(state)
            reward = environment.step(action)
            episode_rewards.append(reward)
            d = environment.is_done()
            steps += 1
            new_state = environment.get_state()
            cache_to_experience_replay(args, experience_replay, state, action, reward, new_state, d)
            states, actions, rewards, new_states, ds = get_batch(args, experience_replay)
            agent.update_q(states=states, actions=actions, rewards=rewards, new_states=new_states, ds=ds)
        if run_experiment:
            experiment.update(steps=steps, rewards=episode_rewards)
            if experiment.is_done():
                break
    if run_experiment:
        experiment.show()
    print('done')


def get_batch(args, experience_replay):
    batch_size = min(args['batch_size'], len(experience_replay))
    indices = np.random.choice(len(experience_replay), batch_size)
    experience_replay_batch = [experience_replay[i] for i in indices]
    states, actions, rewards, new_states, ds = map(list, zip(*experience_replay_batch))
    return states, actions, rewards, new_states, ds


def cache_to_experience_replay(args, experience_replay, state, action, reward, new_state, d):
    experience_replay.append((state, action, reward, new_state, d))
    if len(experience_replay) > args['experience_replay_capacity']:
        experience_replay.pop(np.random.randint(len(experience_replay)))


def q_learning(args):
    agent_class = getattr(import_module('hw1.Agents.' + args['agent']), 'Agent')
    environment_class = getattr(import_module('hw1.Environments.' + args['environment']), 'Environment')
    environment = environment_class()
    agent = agent_class(environment, args)
    run_experiment = args['experiment'] is not None
    experiment = None
    if run_experiment:
        experiment_class = getattr(import_module('hw1.Experiments.' + args['experiment']), 'Experiment')
        experiment = experiment_class(environment, agent, args)
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
            agent.update_q(state=state, action=action, new_q=new_q)
        if run_experiment:
            experiment.update(steps=steps, rewards=rewards)
            if experiment.is_done():
                break
    if run_experiment:
        experiment.show()
    print('done')


if __name__ == '__main__':
    main()