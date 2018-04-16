import numpy as np
import pdb
from scipy import stats

def my_knn_classify(Xtrn, Ctrn, Xtst, Ks):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of labels for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data (dtype=np.float_)
    #   Ks   : List of the numbers of nearest neighbours in Xtrn
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=np.int_)

    N=np.shape(Xtst)[0]
    M=np.shape(Xtrn)[0]
    D=np.shape(Xtrn)[1]

    # DEFINING THE SHAPE OF XX AND YY.
    XX=np.zeros((N,1),dtype=np.float32)
    YY=np.zeros((M,1),dtype=np.float32)

    '''     / X_1 * X_1^T \             / Y_1 * Y_1^T  \
       XX = |      .        | ,   YY = |      .        | ,
            |      .        |          |      .        |
             \ X_N * X_N^T /            \ Y_M * Y_M^T /
    '''


    XX += (Xtst**2).sum(axis=1, keepdims=True)
    YY += (Xtrn**2).sum(axis=1, keepdims=True)

    intermediate = np.zeros((N,M), dtype=np.float32)
    intermediate += np.dot(Xtst, Xtrn.transpose())

    # XX is N x 1 and YY is M x 1 matrix

    DI = ( XX + (-2 * intermediate ) ) + YY.transpose()

    DI_index = np.argsort(DI,kind='quicksort')


    Cpreds=np.zeros((N,len(Ks)))

    #Now, going through every row in DI_index
    #Take the first K elements from the row and find class labels associated with those elements
    #Cpreds is then taken as the mode of the classes labels

    for i in range(0, N):
        nearestK = []
        for j in range(len(Ks)):
            row = DI_index[i]
            nearestK = np.copy(row[:Ks[j]])
            for k in range(Ks[j]):
                nearestK[k] = Ctrn[nearestK[k]]

            Cpreds[i][j] = stats.mode(nearestK)[0][0]

    return Cpreds

def main():
    Xtrn=np.array([[0,2],[0,4],[1,2],[2,3],[2,1],[3,1],[3,3],[4,4]])
    Xtst=np.array([[2,2]])
    Ctrn=np.array([[0],[0],[0],[0],[1],[1],[1],[1]])
    Ks=[3,5]

    cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, Ks)

    print (cpreds)
