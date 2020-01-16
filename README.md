# smina-docking-benchmark

## Environment

The best way is to use conda environment.
Create new environment and run `docking_benchmark/install_conda_env.sh` script.

## Data

In order to run experiments or train models additional data is required.
Download [this zip](https://drive.google.com/open?id=1HJNgHBWE2eZc2gsHQhqay-V17GaviIxQ), unpack it and set the `DOCKING_BENCHMARK_DATA` environment variable to this directory.

## Experiments

Both experiments requires `DOCKING_BASELINES_DIR` environment variable to be set with path to docking_baselines package.

### Modified physics optimization

This experiment "changes the physics" used by smina during docking,
generates molecule that are supposed to dock well according to the custom scoring function
and double evaluates the generated molecules with default scoring function.

The experiment requires `CUSTOM_SCORING_DIR` environment variable to be set with path to directory
containing custom scorings text files

To run it use the `experiments/modified_physics_optimization/run_experiment.sh` script.
The script requires three arguments in given order:
* `model` - `gvae` or `cvae` - pretrained model used for generating molecules
* `dataset` - dataset used to fine-tune the model;
available datasets for a given protein are listed in `$DOCKING_BENCHMARK_DATA/proteins_data/protein/metadata.json` file;
dataset should contain scores for scoring function that will be used for molecule generation
* `custom_scoring_function` - "new physics" that will be used during initial docking;
you may choose a function defined in `experiments/modified_physics_optimization/custom_scorings`,
although remember to pick the corresponding dataset

### Single component optimization

This experiment generates molecules that optimize a single component of docking score
instead of full score and double evaluate the generated molecules using default scoring function.

To run it use the `experiments/single_component_optimization/run_experiment.sh` script.
The script requires four arguments in given order:
* `model` - `gvae` or `cvae` - pretrained model used for generating molecules
* `dataset` - dataset used to fine-tune the model;
available datasets for a given protein are listed in `DOCKING_BENCHMARK_DATA/proteins_data/protein/metadata.json` file;
the dataset defines the component to be optimized by properly setting `score_column` in `metadata` file
* `mode` - `minimize` or `maximize` - whether the model should minimize or maximize the component
* `protein` - protein that ligand will be docked to; possible choices: `5ht1b`, `5ht2b`, `acm2`
