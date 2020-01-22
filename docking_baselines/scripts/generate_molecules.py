import argparse
import logging
import os.path

import docking_benchmark.data.proteins as proteins
from docking_baselines.models.models import ALL_MODELS
from docking_baselines.utils import scripting
from docking_benchmark.utils.logging import setup_and_get_logger, disable_rdkit_logging

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--model_path')
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('-p', '--protein', default='5ht1b')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-n', '--n-molecules', type=int, default=250)
    parser.add_argument('-m', '--mode', default='minimize')
    parser.add_argument('-r', '--random-samples', type=int, default=100)
    parser.add_argument('-f', '--fine-tune-epochs', type=int, default=5)
    parser.add_argument('--dataset', default='default')
    parser.add_argument('--n-cpu', default=4, type=int)

    args = parser.parse_args()
    setup_and_get_logger(args.debug)
    scripting.set_keras_cores(args.n_cpu)
    disable_rdkit_logging()

    if args.model not in ALL_MODELS:
        logger.error(f'No model named {args.model}')
        raise ValueError(f'No model named {args.model}')

    if args.model_path is None:
        if 'pretrained' not in ALL_MODELS[args.model]:
            raise ValueError(f'No pretrained {args.model} model delivered. '
                             'Provide the path to pretrained model.')

        args.model_path = ALL_MODELS[args.model]['pretrained']

    if args.debug:
        args.n_molecules = 2
        args.fine_tune_epochs = 0
        args.random_samples = 2
        logger.debug('Arguments updated due to debug mode on, current args: %s', str(args))

    return args


def generate_and_dock_molecules(*, model, model_path, output_dir, protein, n_molecules, mode, random_samples,
                                fine_tune_epochs, dataset, n_cpu):
    protein = proteins.get_proteins()[protein]
    dataset = protein.datasets[dataset]
    model_cls = ALL_MODELS[model]['cls']
    generator = model_cls(model_path, dataset, mode=mode, fine_tune_epochs=fine_tune_epochs, output_dir=output_dir,
                          docking_n_cpu=n_cpu)
    optimized, random_gauss = generator.generate_optimized_molecules(
        n_molecules,
        smiles_docking_score_fn=protein.dock_smiles_to_protein,
        random_samples=random_samples,
        with_random_samples=random_samples > 0
    )

    optimized.save(os.path.join(output_dir, 'generated.om'))
    random_gauss.save(os.path.join(output_dir, 'gauss.om'))
    optimized.to_csv(os.path.join(output_dir, 'decomposed_optimized_molecules.csv'), without_columns=['latent_vector'])


if __name__ == '__main__':
    args = vars(_parse_args())

    if 'debug' in args:
        del args['debug']

    generate_and_dock_molecules(**args)
