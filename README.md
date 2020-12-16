# We Should at Least Be Able to Design Molecules That Dock Well

![Docking Benchmark Flow](images/docking_benchmark_flow.png?raw=true)

**To learn how to evaluate your model, [see Getting Started notebook](notebooks/getting-started.ipynb).**

Note: We should release in January v2 of the benchmark that adds better baselines and accounts for diversity.

## Results

Listed below are benchmark results from the paper for docking score optimization (the lower, the better). Each cell reports the mean score for all compounds, and for the top 1% of compounds in the parenthesis. ZiNC row contains 250 random molecules from fine-tuning dataset.

|               | 5HT1B             | 5HT2B            | ACM2             | CYP2D6           |
|---------------|-------------------|------------------|------------------|------------------|
| **ZiNC**      | -8.241 (-12.068)  | -8.303 (-14.477) | -7.587 (-11.533) | -6.873 (-10.601) |
| **Inactives** | -7.707 (-11.306)  | -8.375 (-11.212) | -6.971 (-10.451) | -6.992 (-10.76)  |
| **Actives**   | -8.727 (-12.294)  | -8.527 (-14.38)  | -8.156 (-11.532) | -6.866 (-8.869)  |
| **CVAE**      | -4.888 (-8.942)   | -5.349 (-9.767)  | -5.138 (-7.600)  | -4.829 (-7.719)  |
| **GVAE**      | -4.681 (-7.507)   | -4.139 (-6.983)  | -5.156 (-7.869)  | -5.425 (-7.590)  |
| **REINVENT**  | -10.412 (-11.480) | -9.084 (-11.65)  | -6.798 (-9.285)  | -9.145 (-11.463) |

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
