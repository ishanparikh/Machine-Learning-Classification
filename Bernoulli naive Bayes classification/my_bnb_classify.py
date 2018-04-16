import numpy as np
import pdb

def my_bnb_classify(Xtrn, Ctrn, Xtst, threshold):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of label vector for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data matrix (dtype=np.float_)
    #   threshold   : A scalar threshold (type=float)
    # Output:
    #  Cpreds : N-by-1 ndarray of predicted labels for Xtst (dtype=np.int_)

    #YourCode - binarisation of Xtrn and Xtst.

    #YourCode - naive Bayes classification with multivariate Bernoulli distributions

    N=np.shape(Xtst)[0]
    M=np.shape(Xtrn)[0]
    D=np.shape(Xtrn)[1]

    #BINARIZATION
    Xtrn = np.where(Xtrn < threshold, 0,1)
    Xtst = np.where(Xtst < threshold, 0,1)

    Cpreds = np.zeros((N,), dtype=np.float)

    class_list = np.sort(np.unique(Ctrn)) #Takes all unique classes
    num_classes = len(class_list)
    matrix_cl = np.zeros((num_classes, D))
    p_class = np.zeros(num_classes, dtype=np.float)     #Probablity of each class

    for i in range(len(class_list)):

        #pdb.set_trace()
        idx = np.argwhere(Ctrn == class_list[i]).T[0]     #gives index of things belonging to a particular class
        p_class[i] = len(idx)/float(M)
        temp = np.take(Xtrn,idx, axis=0)                  #all rows that belong to particular class
        matrix_cl[i] = np.sum(temp, axis=0)/float(len(idx))     #taking column sum and finding avg
        #pdb.set_trace()


    #applying naive bayes model
    for i in range(N):

        PP = ((np.tile(Xtst[i], (num_classes, 1))) * matrix_cl) + ((np.tile(1-Xtst[i], (num_classes, 1))) * (1-matrix_cl))
        PP = np.prod(PP, axis=1)
        PP = PP * p_class
        Cpreds[i] = class_list[np.argmax(PP)]


    Cpreds = np.transpose(Cpreds)

    return Cpreds


# def main():
#
#     Xtrn = np.array([[1,0,0,0,1,1,1,1],
#                   [0,0,1,0,1,1,0,0],
#                   [0,1,0,1,0,1,1,0],
#                   [1,0,0,1,0,1,0,1],
#                   [1,0,0,0,1,0,1,1],
#                   [0,0,1,1,0,0,1,1],
#                   [0,1,1,0,0,0,1,0],
#                   [1,1,0,1,0,0,1,1],
#                   [0,1,1,0,0,1,0,0],
#                   [0,0,0,0,0,0,0,0],
#                   [0,0,1,0,1,0,1,0]])
#
#
#     Ctrn=np.array([0,0,0,0,0,0,1,1,1,1,1])
#
#     Xtst=np.array([[1,0,0,1,1,1,0,1],
#                    [0,1,1,0,1,0,1,0]])
#
#     my_bnb_classify(Xtrn, Ctrn, Xtst, 1)
#
