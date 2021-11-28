import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal
import os

def dir_creation(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def linear_simulator(X, mcau=30, n=5000, cau=range(30)):
    sigma_g = 0.5
    sigma_eps = 1 - sigma_g
    N = X.shape[0]
    X_cau = scale(X[:,mcau])
    Kin = X_cau@X_cau.T
    for i in range(5):
        y = np.random.multivariate_normal(np.zeros(N), np.eye(N) * sigma_eps + sigma_g/mcau * Kin)
        savepath = f'/u/flashscratch/b/boyang19/CS269/code/Parallel-SHAP/simulations/linear/m_{m}_mcau_{mcau}_n_{n}_sim_{i}.pheno'
        dir_creation(savepath)
        np.savetxt(savepath,y)

def nonlinear_simulator(X, mcau=30, n=5000, cau=range(30)):
    sigma_g = 0.5
    sigma_eps = 1 - sigma_g
    N = X.shape[0]
    X_cau = scale(X[:,mcau])
    Kactual = pairwise_kernels(X_cau, metric='rbf', gamma=0.1)
    for i in range(5):
        y = np.random.multivariate_normal(np.zeros(N), np.eye(N) * sigma_eps + sigma_g * Kactual)
        savepath = f'/u/flashscratch/b/boyang19/CS269/code/Parallel-SHAP/simulations/nonlinear/m_{m}_mcau_{mcau}_n_{n}_sim_{i}.pheno'
        dir_creation(savepath)
        np.savetxt(savepath,y)


if __name__ == "__main__":
    np.random.seed(1)
    n = 5000
    for m in [50, 100, 500, 1000]:
        maf = np.random.uniform(0.05,0.5)
        X = np.random.binomial(2, maf, (n, m))
        print('Done simulating the data')
        geno_path = f'/u/flashscratch/b/boyang19/CS269/code/Parallel-SHAP/simulations/Data/X.txt'
        dir_creation(geno_path)

        linear_simulator(X)
        print('Done simulating the linear effect')

        nonlinear_simulator(X)
        print('Done simulating the nonlinear effect')

    
