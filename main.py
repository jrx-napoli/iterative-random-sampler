import sys

import datasets
import models
from iterative_sampler import IterativeSampler
from options import get_args


def run(args):
    X_train, X_test, y_train, y_test = datasets.__dict__[args.experiment](args=args)
    model = models.__dict__[args.model]()

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
