# We Should at Least Be Able to Design Molecules That Dock Well

![Docking Benchmark Flow](images/docking_benchmark_flow.png?raw=true)

**To learn how to evaluate your model, [see Getting Started notebook](notebooks/getting-started.ipynb).**

Paper: https://arxiv.org/abs/2006.16955.

News: 
  * (12.2020) We have released v1.0 of the benchmark. Main changes include changing the scoring function to a more accurate one (Vinardo), adding stronger baselines (REINVENT and sampling molecules from ZINC), and measuring and ensuring sufficient novelty and diversity of generated molecules.

## Results

Listed below are benchmark results from the paper for docking score optimization (the lower, the better). Each cell reports the mean score for the generated compounds and their internal diversity in parenthesis. For each protein we sampled a set of molecules from ZINC subset of protein's training set size. As a baseline, we also report results for the top 10% molecules from the training set and ZINC. Please see our paper for more details.

|                 | 5HT1B           | 5HT2B          | ACM2           | CYP2D6         |
|-----------------|-----------------|----------------|----------------|----------------|
| **CVAE**        | -4.647 (0.907)  | -4.188 (0.913) | -4.836 (0.905) | -              |
| **GVAE**        | -4.955 (0.901)  | -4.641 (0.887) | -5.422 (0.898) | -7.672 (0.714) |
| **REINVENT**    | -9.774 (0.506)  | -8.657 (0.455) | -9.775 (0.467) | -8.759 (0.626) |
| **Train (10%)** | -10.837 (0.749) | -9.769 (0.831) | -8.976 (0.812) | -9.256 (0.869) |
| **ZINC (10%)**  | -9.894 (0.862)  | -9.228 (0.851) | -8.282 (0.860) | -8.787 (0.853) |

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
