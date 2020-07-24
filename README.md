# We Should at Least Be Able to Design Molecules That Dock Well

![Docking Benchmark Flow](images/docking_benchmark_flow.png?raw=true)

**To learn how to evaluate your model, [see Getting Started notebook](notebooks/getting-started.ipynb).**

## Environment

The best way is to use conda environment.
Create new environment and run `docking_benchmark/install_conda_env.sh` script.

## Data

In order to run experiments or train models additional data is required.
Download [this zip](https://drive.google.com/open?id=1HJNgHBWE2eZc2gsHQhqay-V17GaviIxQ), unpack it and set the `DOCKING_BENCHMARK_DATA` environment variable to this directory.

## Experiments

### Single component optimization

Run the `docking_baselines/scripts/generate_molecules.py` script. Run it with `-h` flag for info about arguments.

Details about some of the arguments:
* `protein` - protein that ligand will be docked to; possible choices: `5ht1b`, `5ht2b`, `acm2`
* `random_samples` - Gauss samples that will be docked (see paper)
* `mode` - `minimize` or `maximize` - whether the model should minimize or maximize the component
* `dataset` - dataset used to fine-tune the model;
available datasets for a given protein are listed in `DOCKING_BENCHMARK_DATA/proteins_data/protein/metadata.json` file;
the dataset defines the component to be optimized by properly setting `score_column` in `metadata` file
