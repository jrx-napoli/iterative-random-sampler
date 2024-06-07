import argparse


def get_args(argv):
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--experiment', type=str, default='classification', choices=['classification', 'regression'],
                        help='Type of the experiment')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed. If defined all random operations will be reproducible')

    # Data
    parser.add_argument('--n_samples', type=int, default=1000, help='Total number of samples in generated dataset')
    parser.add_argument('--n_classes', type=int, default=10, help='Number of classes in generated dataset')
    parser.add_argument('--n_features', type=int, default=20, help='Number of data features in generated dataset')
    parser.add_argument('--n_informative', type=int, default=5,
                        help='Number of informative features in generated dataset')

    # Model
    parser.add_argument('--model', type=str, default='random_forest', help='Name of the model')

    # Sampling
    parser.add_argument('--strategy', type=str, default='hard', help='Iterative sampling strategy')
    parser.add_argument('--init_sample_pool_size', type=float, default=0.2,
                        help='Percentage of training data available in first training iteration')
    parser.add_argument('--step_size', type=int, default=50,
                        help='Number of data points to be sampled in each iteration')
    parser.add_argument('--max_iter', type=int, default=10, help='Total number of training iterations')

    return parser.parse_args(argv)
