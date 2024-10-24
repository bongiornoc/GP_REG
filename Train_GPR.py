import os
import argparse
import pandas as pd
import gplearn.genetic
from utils import append_dataframe_with_lock

def main(file_path, seed,population_size, parsimony_coefficient, generations, n_jobs):
    # Load the data
    data = pd.read_csv(file_path)

    # Initialize the Symbolic Regressor
    gp = gplearn.genetic.SymbolicRegressor(
        population_size=population_size,
        parsimony_coefficient=parsimony_coefficient,
        generations=generations,
        verbose=True,
        n_jobs=n_jobs,
        random_state=seed
    )

    # Fit the model
    gp.fit(data[['r', 'q']], data['KL'])

    # Output the results
    program_str = str(gp._program)
    raw_fitness = gp._program.raw_fitness_
    program_length = gp._program.length_

    # Create a DataFrame to store the results
    results = pd.DataFrame({
        'program': [program_str],
        'raw_fitness': [raw_fitness],
        'program_length': [program_length],
        'seed': [seed]
        })
    
    # Write the results to a file
    append_dataframe_with_lock(file_path.replace('data_','output_'), results)



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Symbolic Regressor on generated data.")
    parser.add_argument('file_path', type=str, help='Path to the data file')
    parser.add_argument('--seed', type=int, default=None, help='Seed value for random number generator')
    parser.add_argument('--population_size', type=int, default=50000, help='Population size for genetic algorithm. Default is 50000')
    parser.add_argument('--parsimony_coefficient', type=float, default=1e-4, help='Parsimony coefficient for genetic algorithm. Default is 1e-4')
    parser.add_argument('--generations', type=int, default=40, help='Number of generations for genetic algorithm. Default is 40')
    parser.add_argument('--n_jobs', type=int, default=os.cpu_count(), help='Number of jobs for genetic algorithm. Default is the number of CPUs available')

    args = parser.parse_args()

    main(args.file_path,args.seed, args.population_size, args.parsimony_coefficient, args.generations, args.n_jobs)