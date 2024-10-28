import os
import scipy.stats as stats
import numpy as np
import scipy.linalg as la
import argparse
import pandas as pd

from utils import append_dataframe_with_lock

"""
GenerateData Class
This class provides methods to generate data using Inverse Wishart and Wishart distributions, 
normalize the data, and compute Kullback-Leibler (KL) divergence.
Methods
-------
__init__(n)
    Initializes the GenerateData class with the given dimension `n`.
_generate_WhiteInvWishart(q_star)
    Generates white noise using the Inverse Wishart distribution.
diag_normalization()
    Normalizes the diagonal elements of the population covariance matrix.
SampleInvWishart(q_stars, size)
    Samples from the Inverse Wishart distribution and normalizes the population covariance matrix.
sample_normal(df, scale)
    Samples from a multivariate normal distribution and computes the covariance matrix.
SampleWishart(q_samples, size)
    Samples from the Wishart distribution or a multivariate normal distribution based on the given parameters.
Compute_KL()
    Computes the Kullback-Leibler (KL) divergence between the population and sample covariance matrices.
compute_r(q_star, q_sample)
    Computes the ratio `r` based on the given `q_star` and `q_sample`.
check_parameters(q, param_type)
    Validates the given parameter range `q` and ensures it is a tuple with valid bounds.
sample_training_data(t, q_star_range, q_sample_range, size)
    Samples training data based on the given ranges for `q_star` and `q_sample`, and the specified size.
Example
-------
    q_star_range = (0.1, 0.9)
    q_sample_range = (0.1, 0.9)
    df = data.sample_training_data(size, q_star_range, q_sample_range, size)
"""


class GenerateData:
    """
    GenerateData is a class for generating data using various statistical methods, including the Inverse Wishart and Wishart distributions. It also provides methods for computing the Kullback-Leibler (KL) divergence and other statistical measures.
    Methods
    -------
    __init__(n)
        Initializes the GenerateData object with a given dimension n.
    _generate_WhiteInvWishart(q_star)
        Generates white noise using the Inverse Wishart distribution.
    diag_normalization()
        Normalizes the diagonal elements of the PopulationCovariance matrix.
    SampleInvWishart(q_stars, size)
        Samples from the Inverse Wishart distribution and normalizes the results.
    sample_normal(df, scale)
        Samples from a multivariate normal distribution and computes the covariance matrix.
    SampleWishart(q_samples, size)
        Samples from the Wishart distribution or a multivariate normal distribution based on the given parameters.
    Compute_KL()
        Computes the Kullback-Leibler (KL) divergence between the PopulationCovariance and SampleCovariance matrices.
    compute_r(q_star, q_sample)
        Computes the ratio r based on the given q_star and q_sample parameters.
    check_parameters(q, param_type)
        Validates the given q parameters and ensures they are within the specified range.
    sample_training_data(t, q_star_range, q_sample_range, size)
        Samples training data based on the given ranges for q_star and q_sample, and computes the KL divergence and other statistical measures.
    Attributes
    ----------
    n : int
        The dimension of the covariance matrices.
    Id : ndarray
        Identity matrix of dimension n.
    PopulationCovariance : ndarray
        Covariance matrix sampled from the Inverse Wishart distribution.
    SampleCovariance : ndarray
        Covariance matrix sampled from the Wishart distribution or a multivariate normal distribution.
    OracleEigVal : ndarray
        Eigenvalues of the PopulationCovariance matrix in the basis of the SampleCovariance eigenvectors.
    oracle_inverse_covariance : ndarray
        Inverse of the PopulationCovariance matrix in the basis of the SampleCovariance eigenvectors.
    """

    def __init__(self,n,size,seed=None) -> None:
        """
        Initializes the class with the given number of dimensions.
        Parameters:
            n (int): The number of dimensions for the identity matrix.
            seed (int): The seed value for the random number generator.
        Attributes:
            n (int): Stores the number of dimensions.
            Id (numpy.ndarray): An identity matrix of size n x n.
        """
        if size <= 1:
            raise ValueError("size should be greater than 1")
        if isinstance(size,int) == False:
            raise ValueError("size should be an integer")

        self.n = n
        self.Id = np.eye(n)
        self.size = size
        self.rng = np.random.default_rng(seed)

    def Sample_WhiteInvWishart(self,q_star):
        """
        Generate a sample from the inverse Wishart distribution scaled by a factor.
        Parameters:
        -----------
        q_star : float
            A scaling factor used to compute the degrees of freedom for the inverse Wishart distribution.
        Returns:
        --------
        numpy.ndarray
            A sample from the inverse Wishart distribution, scaled by (t_star - self.n - 1).
        Notes:
        ------
        The degrees of freedom for the inverse Wishart distribution is computed as `t_star = self.n / q_star`.
        The sample is then scaled by the factor `(t_star - self.n - 1)`.
        """

        t_star = self.n/q_star

        # Generate the white noise  
        self.PopulationCovariance = stats.invwishart.rvs(df = t_star, 
                                                         scale = self.Id, 
                                                         size = self.size,
                                                         random_state=self.rng)
        
        self.PopulationCovariance = self.diag_normalization(self.PopulationCovariance)
    
    def diag_normalization(self,matrix):
        """
        Normalize the diagonal elements of the given matrix.

        Parameters:
        -----------
        matrix : np.ndarray
            The matrix to normalize.
        Returns:
        --------
        np.ndarray
            The matrix with normalized diagonal elements.
        """

        diag = np.expand_dims( np.sqrt(matrix.diagonal(axis1=1,axis2=2)) ,-1)
        return matrix/ (diag*diag.transpose(0,2,1))
    
    
    def sample_covariance(self,df, scale):
        """
        Generates a sample covariance matrix from a multivariate normal distribution.
        Parameters:
        df (int): Degrees of freedom for the sample.
        scale (array_like): Covariance matrix of the distribution.
        Returns:
        numpy.ndarray: Covariance matrix of the sampled data.
        """
        factor =  np.vectorize(la.cholesky, signature='(n,n)->(n,n)')( self.PopulationCovariance )
        factor = factor.transpose(0,2,1)

        x = factor @ self.rng.normal(size=(self.size,self.n,df))

        return x @ x.transpose(0,2,1)/df
    
    def SampleWishart(self,q_sample):
        """
        Generates samples from a Wishart distribution based on the provided parameters.
        Parameters:
        -----------
        q_samples : array-like
            An array of q samples used to compute t_samples.
        size : int
            The number of samples to generate.
        Returns:
        --------
        None
            The function updates the `SampleCovariance` attribute of the class with the generated samples.
        Notes:
        ------
        - The function computes t_samples as the ratio of `self.n` to `q_samples`.
        - It creates a mask `high_dim_mask` to identify elements where t_samples is greater than `self.n`.
        - For elements where `high_dim_mask` is True, samples are drawn from a Wishart distribution.
        - For elements where `high_dim_mask` is False, samples are drawn using the `sample_normal` method.
        - The generated samples are stored in the `SampleCovariance` attribute of the class.
        """

        t_sample = int(round(self.n/q_sample,0))
        
        

        if t_sample > self.n:
            
            wishart_fun =  lambda scale: stats.wishart.rvs(df = t_sample, scale = scale, random_state=self.rng)/t_sample

            self.SampleCovariance = np.vectorize(wishart_fun,
                                                 signature='(n,n)->(n,n)')(scale=self.PopulationCovariance)
            
        else:
            self.SampleCovariance = self.generate_sample_covariance(t_sample,self.PopulationCovariance)

        #self.SampleCovariance = self.diag_normalization(self.SampleCovariance)

        return 


    def Compute_KL(self):
        """
        Compute the Kullback-Leibler (KL) divergence between the population covariance matrix 
        and the Oracle Estimator.
        This method calculates the KL divergence by performing the following steps:
        1. Compute the eigenvalues of the population covariance matrix.
        2. Compute the eigenvectors of the sample covariance matrix.
        3. Calculate the oracle eigenvalues by projecting the population covariance matrix 
           onto the sample eigenvectors.
        4. Compute the normalized difference between the population eigenvalues and the oracle eigenvalues.
        5. Calculate the oracle inverse covariance matrix.
        6. Compute the normalized trace of the product of the oracle inverse covariance matrix 
           and the population covariance matrix.
        7. Calculate the KL divergence using the normalized trace and the normalized eigenvalue difference.
        Returns:
            np.ndarray: The KL divergence for each sample.
        """

        PopulationEigenvalues = np.linalg.eigvalsh(self.PopulationCovariance)
        _,SampleEigenVectors = np.linalg.eigh(self.SampleCovariance)

        self.OracleEigVal = SampleEigenVectors.transpose(0,2,1) @ self.PopulationCovariance @ SampleEigenVectors
        self.OracleEigVal = self.OracleEigVal.diagonal(axis1=1,axis2=2)

        norm_EigVal_logDiff = (np.log(self.OracleEigVal)-np.log(PopulationEigenvalues)).mean(axis=-1)

        self.OracleEigVal = np.expand_dims(self.OracleEigVal,-2)

        self.oracle_inverse_covariance = SampleEigenVectors * 1/self.OracleEigVal @ SampleEigenVectors.transpose(0,2,1)

        norm_trace = (self.oracle_inverse_covariance*self.PopulationCovariance).sum(axis=-1).mean(axis=-1)

        KL = (norm_trace + norm_EigVal_logDiff -1)/2

        return KL
    
    def compute_r(self,q_star,q_sample):
        """
        Compute the ratio r based on the given q_star and q_sample values.
        Parameters:
        q_star (float): The q_star value.
        q_sample (float): The q_sample value.
        Returns:
        float: The computed ratio r.
        """


        numerator = self.n * q_star
        denominator =  self.n * (q_star + q_sample - q_star*q_sample) - q_star*q_star

        return numerator/denominator
    
    def sample_training_data(self,q_star,q_sample):
        
        """
        Samples training data based on given parameters and returns a DataFrame.
        Parameters:
        q_star_range (tuple): A tuple containing the minimum and maximum values for q_star.
        q_sample_range (tuple): A tuple containing the minimum and maximum values for q_sample.
        size (int): The number of samples to generate. Must be a positive integer.
        Returns:
        pd.DataFrame: A DataFrame containing the sampled data with columns:
            - 'r': Computed r values.
            - 'p': Computed p values.
            - 'KL': Computed KL divergence values.
        Raises:
        ValueError: If size is less than 1 or not an integer.
        """

        self.Sample_WhiteInvWishart(q_star)

        self.SampleWishart(q_sample)

        KL = self.Compute_KL()

        r = self.compute_r(q_star,q_sample)

        df = pd.DataFrame({"r":r,"q":q_sample,"KL":KL})

        return df.mean()




if __name__ == "__main__":

    # create the data directory if it does not exist
    if not os.path.exists("data"):
        os.makedirs("data")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate data using Inverse Wishart and Wishart distributions.")
    parser.add_argument('--seed', type=int, default=None, help='Seed value for random number generator')

    parser.add_argument('--n', type=int, default=1000, help='Number of dimensions for the identity matrix')
    parser.add_argument('--repetitions', type=int, default=100, help='Number of repetitions for generating data')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--q_sample_max', type=float, default=1.0, help='Maximum value for q_sample range')
    parser.add_argument('--q_star_max', type=float, default=0.95, help='Maximum value for q_star range')

    args = parser.parse_args()

    seed_value = args.seed
    n = args.n
    repetitions = args.repetitions
    n_samples = args.n_samples
    q_sample_max = args.q_sample_max
    q_star_max = args.q_star_max

    q_star_range = (0,q_star_max)
    q_sample_range = (0.,q_sample_max)

    rng = np.random.default_rng(seed_value)
    q_stars = rng.uniform(*q_star_range, n_samples)
    q_samples = rng.uniform(*q_sample_range, n_samples)

    data = GenerateData(n, repetitions, seed=seed_value)
    df = pd.DataFrame([data.sample_training_data(q_star, q_sample) 
                       for q_star, q_sample in zip(q_stars, q_samples)])

    file_path = f"datasets/input_data_n_{n}_q_sample_max_{q_sample_max:.1f}_q_star_max_{q_star_max:.2f}.csv".replace('.', '_')

    append_dataframe_with_lock(file_path, df)
    print(df)
