# Quantifying Information Loss in Optimal Covariance Matrix Cleaning

This project provides scripts and utilities to generate synthetic data, perform symbolic regression, and analyze the results in the context of quantifying the information lost in optimal covariance matrix cleaning. The code aims to reproduce the experiments and findings from the paper:

Bongiorno, Christian, and Lamia Lamrani. "Quantifying the information lost in optimal covariance matrix cleaning." arXiv preprint arXiv:2310.01963 (2023).

If you use this code in your research, please consider citing the paper.

## Repository Structure

- 

Generate_data.py

: Script to generate synthetic data using Inverse Wishart and Wishart distributions.
- 

Generate_data.sh

: Shell script to submit data generation jobs to a computing cluster.
- 

Train_GPR.py

: Script to train a symbolic regression model using Genetic Programming.
- 

Train_GPR.sh

: Shell script to submit training jobs to a computing cluster.
- 

utils.py

: Utility functions used across the project.
- 

Analysis.ipynb

: Jupyter Notebook for analyzing the symbolic regression results.
- 

requirements.txt

: List of Python dependencies.
- 

.gitignore

: Specifies files and directories to ignore in version control.
- `logs/`: Directory for log files (ignored by version control).

## Installation

### Prerequisites

- Python 3.7 or higher
- `pip` package manager
- Access to a computing cluster with SLURM (for job submission scripts)

### Installing Dependencies

Install the required packages using:

```sh
pip install -r requirements.txt
```

## Usage

### Generating Data

You can generate synthetic data by running 

Generate_data.py

 directly or via the SLURM submission script 

Generate_data.sh

.

#### Running Locally

```sh
python Generate_data.py --n 1000 --repetitions 500 --n_samples 10 --q_sample_max 1.0 --q_star_max 1.0 --seed 42
```

#### Submitting as a SLURM Job

```sh
bash Generate_data.sh
```

### Training the Symbolic Regressor

Train the symbolic regression model using 

Train_GPR.py

.

#### Running Locally

```sh
python Train_GPR.py data/input_data_n_1000_q_sample_max_1_0_q_star_max_1_00.csv --seed 42 --n_jobs 4
```

#### Submitting as a SLURM Job

```sh
bash Train_GPR.sh
```

### Analysis

Analyze the results using 

Analysis.ipynb

:

```sh
jupyter notebook Analysis.ipynb
```

Within the notebook, you can view the best program:

```python
regression_output.loc[0, 'best_program']
```

## Utility Functions

- 

append_dataframe_with_lock

: Safely append a DataFrame to a CSV file with file locking.
- 

SymbolicExpressionConverter

: Convert a symbolic expression string into a 

sympy

 expression.
- 

verify_terms_in_program

: Verify the presence of specific terms in symbolic expressions within a CSV file.

## Data Generation Details

The 

GenerateData

 class in 

Generate_data.py

 includes methods to:

- Sample from the Inverse Wishart distribution: 

Sample_WhiteInvWishart


- Sample from the Wishart distribution: 

SampleWishart


- Compute the Kullback-Leibler divergence: 

Compute_KL


- Compute the ratio 

r

: 

compute_r


- Sample training data: 

sample_training_data



## Citing This Work

Please cite the following paper:

```bibtex
@article{bongiorno2023quantifying,
  title={Quantifying the information lost in optimal covariance matrix cleaning},
  author={Bongiorno, Christian and Lamia Lamrani},
  journal={arXiv preprint arXiv:2310.01963},
  year={2023}
}
```