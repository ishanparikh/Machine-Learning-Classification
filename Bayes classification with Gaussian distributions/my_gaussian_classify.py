import numpy as np
from logdet import *
import pdb

def my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon):
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
    Ms = np.zeros((D,K), dtype=np.float)
    Covs = np.zeros((D,D,K), dtype=np.float)
    Cpreds = np.zeros((N,), dtype=np.float)


    class_list = np.sort(np.unique(Ctrn))
    num_classes = len(class_list)

    for i in range(len(class_list)):
        idx = np.argwhere(Ctrn == class_list[i]).T[0]   #gives index of things belonging to a particular class
        temp = np.take(Xtrn, idx, axis=0)               #all rows that belong to particular class
        matrix_mean = np.sum(temp, axis=0)/float(len(idx)) #helper to create Ms
        Ms[:,i]= matrix_mean
        mm = temp - np.tile(matrix_mean, (len(idx),1)) # xtrn -mean
        Covs[:,:,i] =(1.0/len(idx)) * np.dot((mm).T, mm)
        Covs[:,:,i] = Covs[:,:,i] + (np.identity(D) * epsilon) #adding epsilon to the matrix

    pmat = np.zeros((N,K))

    for i in range(K):

        invcov = np.linalg.inv(Covs[:, :, i]) #finding the inverse of Cov matrix
        mu = np.transpose(Ms[:, i]) #
        #pdb.set_trace()

        Xtst_new = Xtst - np.ones((N, 1)) * mu # xtst - mean
        fact = np.sum(np.dot(Xtst_new, invcov) * Xtst_new, 1)
        pmat[:,i] = (-.5 * fact) + (-0.5 * logdet(Covs[:, :, i])) #finding log posterior probability
        #pdb.set_trace()

    for i in range (N):
        Cpreds[i] = class_list[np.argmax(pmat[i,:])]

    return Cpreds, Ms, Covs



