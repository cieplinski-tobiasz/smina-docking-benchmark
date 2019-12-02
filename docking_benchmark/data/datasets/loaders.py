import os

from docking_benchmark.data.directories import DATASETS

DATASET_SUFFIX = '.smiles'


def _get_lines_in_file(path):
    return sum(1 for _ in open(path))


class Dataset:
    def __init__(self, name):
        self._directory = os.path.join(DATASETS, name)

        if not os.path.exists(self._directory):
            raise ValueError(f'No dataset named {name}')

        split_names = [
            os.path.splitext(os.path.basename(split))[0]
            for split in os.listdir(self._directory)
            if split.endswith(DATASET_SUFFIX)
        ]

        self._splits = {
            split: {
                'path': os.path.abspath(os.path.join(DATASETS, name, split + DATASET_SUFFIX)),
                'size': _get_lines_in_file(os.path.abspath(os.path.join(DATASETS, name, split + DATASET_SUFFIX)))
            }
            for split in split_names
        }

    def size_for_split(self, split):
        return self._splits[split]['size']

    def load_split_as_generator(self, split, *, smiles_to_input_fn, batch_size=64):
        with open(self._splits[split]['path']) as split_file:
            while True:
                batch = []

                while len(batch) < batch_size:
                    line = split_file.readline()

                    if not line:
                        split_file.seek(0)
                        continue

                    batch.append(line.strip())

                model_input = smiles_to_input_fn(batch)
                yield model_input, model_input

    def load_split(self, split, smiles_to_input_fn=None):
        with open(self._splits[split]['path']) as data_file:
            smiles = [line.strip() for line in data_file.readlines() if line]

            if smiles_to_input_fn is not None:
                return smiles_to_input_fn(smiles)

            return smiles
