#! /usr/bin/env python
# A sample template for my_gaussian_system.py

import numpy as np
import scipy.io
from my_gaussian_classify import *
from my_confusion import *
import  pdb
from time import time

# Load the data set
#   NB: replace <UUN> with your actual UUN.
filename = "/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1674417/data.mat";
data = scipy.io.loadmat(filename);

# Feature vectors: Convert uint8 to double, and divide by 255
Xtrn = data['dataset']['train'][0,0]['images'][0,0].astype(dtype=np.float_) /255.0
Xtst = data['dataset']['test'][0,0]['images'][0,0].astype(dtype=np.float_) /255.0
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0,0]['labels'][0,0].astype(dtype=np.int_)-1
Ctst = data['dataset']['test'][0,0]['labels'][0,0].astype(dtype=np.int_)-1

#YourCode - Prepare measuring time

# Run classification
epsilon = 0.01

t=time()
(Cpreds, Ms, Covs) = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon)

#YourCode - Measure the user time taken, and display it.
print('Time elapsed is : ' + str(time()-t))


#YourCode - Get a confusion matrix and accuracy
#YourCode - Save the confusion matrix as "Task3/cm.mat".
cm= my_confusion(Ctst, Cpreds)
scipy.io.savemat('cm.mat', mdict={'cm': cm})


#YourCode - Save the mean vector and covariance matrix for class 26,
#           i.e. save Mu(:,25) and Cov(:,:,25) as "Task3/m26.mat" and
#           "Task3/cov26.mat", respectively.
m26 = Ms[:, 25]
cov26 = Covs[:, :, 25]
scipy.io.savemat('m26.mat', mdict={'m26': m26})
scipy.io.savemat('cov26.mat', mdict={'cov26': cov26})


#YourCode - Display the required information - N, Nerrs, acc.
print(' N = ' + str(len(Xtst)) + ' Nerrs = ' +  str(len(Cpreds) - np.trace(cm[0])) + ' Accuracy = '+ str(cm[1]))







