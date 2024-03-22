import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser for MEME')

    # Environment parameters
    parser.add_argument('--env_name', type=str, default='BreakoutDeterministic-v4',
                        help='Name of the environment')
    parser.add_argument('--num_actors', type=int, default=2,
                        help='Number of actors')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--buffer_size', type=float, default=5e4,
                        help='Buffer size')
    parser.add_argument('--burnin', type=int, default=3,
                        help='Burn-in period')
    parser.add_argument('--rollout', type=int, default=10,
                        help='Rollout length')

    # Algorithm parameters
    parser.add_argument('--epsilon', type=float, default=1,
                        help='Exploration rate')
    parser.add_argument('--epsilon_min', type=float, default=0.1,
                        help='Minimum exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.0001,
                        help='Exploration decay rate')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.999),
                        help='Adam betas')
    parser.add_argument('--adam_eps', type=float, default=1e-8,
                        help='Adam epsilon')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Beta parameter')
    parser.add_argument('--discount_max', type=float, default=0.997,
                        help='Maximum discount factor')
    parser.add_argument('--discount_min', type=float, default=0.99,
                        help='Minimum discount factor')
    parser.add_argument('--tau', type=float, default=0.25,
                        help='Tau parameter')
    parser.add_argument('--update_every', type=int, default=1,
                        help='Update frequency')
    parser.add_argument('--save_every', type=int, default=400,
                        help='Save frequency')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    # Bandit parameters
    parser.add_argument('--bandit_window_size', type=int, default=90,
                        help='Bandit window size')
    parser.add_argument('--bandit_beta', type=float, default=1.0,
                        help='Bandit beta parameter')
    parser.add_argument('--bandit_epsilon', type=float, default=0.5,
                        help='Bandit epsilon parameter')

    # Other parameters
    parser.add_argument('--e', type=float, default=0.01,
                        help='E parameter')
    parser.add_argument('--p_a', type=float, default=0.6,
                        help='P_a parameter')
    parser.add_argument('--p_beta', type=float, default=0.4,
                        help='P_beta parameter')
    parser.add_argument('--p_beta_increment_per_sampling', type=float, default=0.001,
                        help='P_beta increment per sampling')

    # Additional parameters
    parser.add_argument('--num_envs', type=int, help='Number of environments')
    parser.add_argument('--action_size', type=int, help='Size of action space')
    parser.add_argument('--N', type=int, default=2,
                        help='N parameter')
    parser.add_argument('--kernel_epsilon', type=float, default=0.0001,
                        help='Kernel epsilon parameter')
    parser.add_argument('--cluster_distance', type=float, default=0.008,
                        help='Cluster distance parameter')
    parser.add_argument('--max_similarity', type=float, default=8.0,
                        help='Maximum similarity parameter')
    parser.add_argument('--c_constant', type=float, default=0.001,
                        help='C constant parameter')

    args = parser.parse_args()

    print(args)
    
    return args

if __name__ == '__main__':
    parse_args()
