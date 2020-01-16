import argparse
import os
from datetime import datetime

from docking_baselines.datasets.loaders import Dataset
from docking_baselines.models.models import ALL_MODELS
from docking_baselines.utils.scripting import set_keras_cores
from docking_benchmark.data.directories import PRETRAINED_MODELS
from docking_benchmark.utils.logging import setup_and_get_logger

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
            'gvae-' + arguments.dataset + '-' + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        )

    return arguments


if __name__ == '__main__':
    args = _parse_args()
    gvae = ALL_MODELS['gvae']['training']['create_fn']()
    dataset = Dataset(args.dataset)

    gvae = ALL_MODELS['gvae']['training']['train_fn'](
        gvae,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mode=args.mode,
        save_path=args.save_path
    )

    test = dataset.load_split('test')
    test_one_hots = gvae.to_one_hots(test)
    test_loss, test_accuracy = gvae.vae.autoencoder.evaluate(test_one_hots, test_one_hots)
    logger.info(f'[TEST] loss: {test_loss:.4f} accuracy: {test_accuracy:.4f}')
