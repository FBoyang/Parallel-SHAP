import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal
import os
from sklearn.metrics.pairwise import pairwise_kernels

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
    for i in range(3):
        y = np.random.multivariate_normal(np.zeros(N), np.eye(N) * sigma_eps + sigma_g/mcau * Kin)
        savepath = f'/u/flashscratch/b/boyang19/CS269/code/Parallel-SHAP/simulations/linear/m{m}/mcau_{mcau}_n_{n}_sim_{i}.pheno'
        dir_creation(savepath)
        np.savetxt(savepath,y)


def fix_linear_simulator(X, mcau=30, n=5000, cau=range(30)):
    sigma_g = 0.5
    sigma_eps = 1 - sigma_g
    N = X.shape[0]
    X_cau = scale(X[:,mcau])
    Kin = X_cau@X_cau.T
    
    for i in range(3):
        beta = np.random.randn(mcau)*sigma_g/mcau
        y = X_cau@beta.T
        savepath = f'/u/flashscratch/b/boyang19/CS269/code/Parallel-SHAP/simulations/linear/m{m}/mcau_{mcau}_n_{n}_sim_{i}.pheno'
        dir_creation(savepath)
        saveparam = f'/u/flashscratch/b/boyang19/CS269/code/Parallel-SHAP/simulations/linear/m{m}/mcau_{mcau}_n_{n}_sim_{i}.beta'
        np.savetxt(savepath,y)
        np.savetxt(saveparam,beta)


def nonlinear_simulator(X, mcau=30, n=5000, cau=range(30)):
    sigma_g = 0.5
    sigma_eps = 1 - sigma_g
    N = X.shape[0]
    X_cau = scale(X[:,mcau])
    Kactual = pairwise_kernels(X_cau, metric='rbf', gamma=0.1)
    for i in range(3):
        y = np.random.multivariate_normal(np.zeros(N), np.eye(N) * sigma_eps + sigma_g * Kactual)
        savepath = f'/u/flashscratch/b/boyang19/CS269/code/Parallel-SHAP/simulations/nonlinear/m{m}/_mcau_{mcau}_n_{n}_sim_{i}.pheno'
        dir_creation(savepath)
        np.savetxt(savepath,y)


if __name__ == "__main__":
    np.random.seed(1)
    n = 5000
    for m in [50, 100, 500, 1000]:
        maf = np.random.uniform(0.05,0.5)
        X = np.random.binomial(2, maf, (n, m))
        print('Done simulating the data')
        geno_path = f'/u/flashscratch/b/boyang19/CS269/code/Parallel-SHAP/simulations/Data/X_{m}.txt'
        dir_creation(geno_path)
        np.savetxt(geno_path,X)
        print('Done saving the data')

        fix_linear_simulator(X)
        print('Done simulating the linear effect')

        nonlinear_simulator(X)
        print('Done simulating the nonlinear effect')

    
