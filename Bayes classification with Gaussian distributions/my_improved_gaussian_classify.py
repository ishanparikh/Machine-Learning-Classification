import numpy as np
from logdet import *
import pdb
import  scipy
import matplotlib

def my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon,sz):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of label vector for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data matrix (dtype=np.float_)
    #   epsilon   : A scalar parameter for regularisation (type=float)
    # Output:
    #  Cpreds : N-by-1 ndarray of predicted labels for Xtst (dtype=int_)
    #  Ms    : D-by-K ndarray of mean vectors (dtype=np.float_)
    #  Covs  : D-by-D-by-K ndarray of covariance matrices (dtype=np.float_)

    #YourCode - Bayes classification with multivariate Gaussian distributions.

    N=np.shape(Xtst)[0]
    M=np.shape(Xtrn)[0]
    D=np.shape(Xtrn)[1]
    K=26

    Ms = np.zeros((sz,K), dtype=np.float)
    Covs = np.zeros((sz,sz,K), dtype=np.float) #changed the size acc to PCA dimension
    Cpreds = np.zeros((N,), dtype=np.float)


    class_list = np.sort(np.unique(Ctrn))

    matrix_mean = np.sum(Xtrn, axis=0)/float(N)


    Covs0=(1.0/(N) * np.dot((Xtrn-matrix_mean).T, Xtrn-matrix_mean))
    Covs0 = Covs0 + (np.identity(D) * epsilon)

    pmat = np.zeros((N,K))

    #finding eigen values and corresponding eigen vectors
    eig_val_cov, eig_vec_cov = np.linalg.eigh(Covs0)
    eig_vec_cov = eig_vec_cov[:, ::-1]

    #choosing the eigen vectors up to the right PCA dimention
    chosen_vec = eig_vec_cov[:, :sz]

    new_trn = np.dot(Xtrn, chosen_vec)

    #applying classification with a single Gaussian distribution as before
    for i in range(len(class_list)):
        idx = np.argwhere(Ctrn == class_list[i]).T[0]

        temp = np.take(new_trn, idx, axis=0)

        matrix_mean = np.sum(temp, axis=0)/float(len(idx))

        Ms[:,i]= matrix_mean
        mm = temp - np.tile(matrix_mean, (len(idx),1))
        Covs[:,:,i] =(1.0/len(idx)) * np.dot((mm).T, mm)
        Covs[:,:,i] = Covs[:,:,i] + (np.identity(sz) * epsilon)

    Xtst = Xtst.dot(chosen_vec)
    for i in range(K):
        invcov = np.linalg.inv(Covs[:, :, i])
        mu = np.transpose(Ms[:, i])

        Xtst_new = Xtst - mu
        fact = np.sum(np.dot(Xtst_new, invcov) * Xtst_new, 1)
        pmat[:,i] = (-.5 * fact) + (-0.5 * logdet(Covs[:, :, i]))


    for i in range (N):
        Cpreds[i]=class_list[np.argmax(pmat[i,:])]

    return (Cpreds, Ms, Covs)

