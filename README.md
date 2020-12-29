# We Should at Least Be Able to Design Molecules That Dock Well

![Docking Benchmark Flow](images/docking_benchmark_flow.png?raw=true)

**To learn how to evaluate your model, [see Getting Started notebook](notebooks/getting-started.ipynb).**

Paper: https://arxiv.org/abs/2006.16955.

News: 
  * (12.2020) We have released v1.0 of the benchmark. We have changed scoring function to Vinardo based on feedback we received. We also added novelty filter to ensure that generated compounds are not too similar to the training set. We report diversity on top of score achieved in each benchmark. Finally, we added REINVENT.

## Results

Listed below are benchmark results from the paper for docking score optimization (the lower, the better). Each cell reports the mean score for the generated compounds. For ZINC, we sampled 1000 compounds and report the top 10% docking score. Please see paper for more details.

|               | 5HT1B             | 5HT2B            | ACM2             | CYP2D6           |
|---------------|-------------------|------------------|------------------|------------------|
| **CVAE**      | -4.647 | -4.188  |  -4.836  | -  |
| **GVAE**      | -4.955   | -4.641  | -5.422  | -7.672  |
| **REINVENT**  | -9.774  | -8.657  | -9.775   | -8.759 |
| **ZINC (top 10%)** | -9.894   | -9.228 | -8.282 | -8.787 |

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
