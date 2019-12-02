import argparse
import os
from datetime import datetime

from docking_benchmark.data.datasets.loaders import Dataset
from docking_benchmark.data.directories import PRETRAINED_MODELS
from docking_benchmark.models.models import ALL_MODELS
from docking_benchmark.utils.scripting import setup_and_get_logger, set_keras_cores

logger = setup_and_get_logger(True, __name__)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-m', '--mode', default='generator')
    parser.add_argument('--n-cpu', type=int, default=4)
    parser.add_argument('-s', '--save-path')
    arguments = parser.parse_args()
    set_keras_cores(arguments.n_cpu)

    if arguments.save_path is None:
        arguments.save_path = os.path.join(
            PRETRAINED_MODELS,
            'cvae-' + arguments.dataset + '-' + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        )

    return arguments


if __name__ == '__main__':
    args = _parse_args()
    cvae = ALL_MODELS['cvae']['training']['create_fn']()
    dataset = Dataset(args.dataset)

    for epoch in range(1, args.epochs + 1):
        cvae = ALL_MODELS['cvae']['training']['train_fn'](
            cvae,
            dataset,
            epochs=1,
            batch_size=args.batch_size,
            mode=args.mode,
        )

        cvae.enc.save(args.save_path + 'epoch-' + str(epoch) + '-encoder')
        cvae.dec.save(args.save_path + 'epoch-' + str(epoch) + '-decoder')
