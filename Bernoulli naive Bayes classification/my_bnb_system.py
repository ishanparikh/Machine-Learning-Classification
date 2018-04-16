#! /usr/bin/env python
# A sample template for my_bnb_system.py

import numpy as np
import scipy.io
from my_bnb_classify import *
from time import time
from my_confusion import *
import matplotlib.pyplot as plt
import pdb
# Load the data set
#   NB: replace <UUN> with your actual UUN.
filename = "/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1674417/data.mat";
data = scipy.io.loadmat(filename);

# Feature vectors: Convert uint8 to double   (but do not divide by 255)
Xtrn = data['dataset']['train'][0,0]['images'][0,0].astype(dtype=np.float_)
Xtst = data['dataset']['test'][0,0]['images'][0,0].astype(dtype=np.float_)
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0,0]['labels'][0,0].astype(dtype=np.int_)-1
Ctst = data['dataset']['test'][0,0]['labels'][0,0].astype(dtype=np.int_)-1



# Run classification
threshold = 1.0

t = time()
Cpreds = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold)

#YourCode - Prepare measuring time
print('time elapsed is = ' + str(time()-t)+' seconds')

#YourCode - Get a confusion matrix and accuracy

#YourCode - Save the confusion matrix as "Task2/cm.mat".
cm = my_confusion(Ctst,Cpreds.T)
scipy.io.savemat('cm.mat', mdict={'cm':cm})


#YourCode - Display the required information - N, Nerrs, acc.

print('N = ' + str(len(Xtst)) + ' Nerrs = ' +  str(len(Cpreds) - np.trace(cm[0])) + ' Accuracy = '+ str(cm[1]))





