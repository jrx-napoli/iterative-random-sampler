import sys

from sklearn.ensemble import RandomForestClassifier

from dataset_gen import make_classification_dataset
from iterative_sampler import IterativeSampler
from options import get_args


def run(args):
    X_train, X_test, y_train, y_test = make_classification_dataset(args=args)

    model = RandomForestClassifier()

    sampler = IterativeSampler(model=model,
                               strategy=args.strategy,
                               init_sample_pool_size=args.init_sample_pool_size,
                               step_size=args.step_size,
                               max_iter=args.max_iter,
                               random_state=args.seed)
    sampler = sampler.fit(X_train, y_train)

    history = sampler.get_history()
    print("Training accuracy:", history)


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    run(args)
