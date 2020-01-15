import argparse
import logging

import docking_benchmark.data.proteins as proteins
from docking_benchmark.docking.smina.docking import dock_smiles
from docking_benchmark.models.models import ALL_MODELS
from docking_benchmark.utils import scripting

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--model_path')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-p', '--protein', default='5ht1b')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-n', '--n-molecules', default=250)
    parser.add_argument('-m', '--mode', default='minimize')
    parser.add_argument('--dataset', default='default')
    parser.add_argument('--n-cpu', default=4, type=int)

    args = parser.parse_args()
    scripting.setup_and_get_logger(args.debug)
    scripting.set_keras_cores(args.n_cpu)
    scripting.disable_rdkit_logging()

    if args.model not in ALL_MODELS:
        logger.error(f'No model named {args.model}')
        raise ValueError(f'No model named {args.model}')

    if args.model_path is None:
        if 'pretrained' not in ALL_MODELS[args.model]:
            raise ValueError(f'No pretrained {args.model} model delivered. '
                             'Provide the path to pretrained model.')

        args.model_path = ALL_MODELS[args.model]['pretrained']

    if args.debug:
        debug_n_molecules = 5
        logger.debug(f'{debug_n_molecules} molecules will be generated '
                     'due to debug mode on.')
        args.n_molecules = debug_n_molecules

    return args


if __name__ == '__main__':
    args = _parse_args()
    protein = proteins.get_proteins()[args.protein]
    dataset = protein.datasets[args.dataset]
    model_cls = ALL_MODELS[args.model]['cls']
    generator = model_cls(args.model_path, dataset, mode=args.mode)
    optimized, random_gauss = generator.generate_optimized_molecules(
        args.n_molecules,
        smiles_docking_score_fn=lambda smi: dock_smiles(smi, protein.path, protein.pocket_center, ),
        random_samples=100,
        with_random_samples=True
    )

    optimized.save(args.output + '.om')
    random_gauss.save(args.output + '.gauss.om')
    optimized.to_csv(args.output, columns=['predicted_score'])
