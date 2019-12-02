import os

DATA = os.environ.get('DOCKING_BENCHMARK_DATA', os.path.dirname(__file__))
PRETRAINED_MODELS = os.path.join(DATA, 'pretrained_models')
PROTEINS = os.path.join(DATA, 'proteins_data')
DATASETS = os.path.join(DATA, 'datasets')
