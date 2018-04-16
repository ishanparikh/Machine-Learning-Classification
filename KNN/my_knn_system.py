#! /usr/bin/env python
# A sample template for my_knn_system.py

import numpy as np
import scipy.io
from my_knn_classify import *
import my_knn_classify as knnc

from my_confusion import *
from time import time

import matplotlib.pyplot as plt

# Load the data set
#   NB: replace <UUN> with your actual UUN.
filename = "/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1674417/data.mat";
data = scipy.io.loadmat(filename);

# Feature vectors: Convert uint8 to double, and divide by 255.
Xtrn = data['dataset']['train'][0,0]['images'][0,0].astype(dtype=np.float_) /255.0
Xtst = data['dataset']['test'][0,0]['images'][0,0].astype(dtype=np.float_) /255.0
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0,0]['labels'][0,0].astype(dtype=np.int_)-1
Ctst = data['dataset']['test'][0,0]['labels'][0,0].astype(dtype=np.int_)-1

#print np.shape(Ctrn)
#YourCode - Prepare measuring time

# Run K-NN classification
kb = [1,3,5,10,20];
#Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, kb)

# YourCode - Measure the user time taken, and display it.
t = time()
Cpreds=my_knn_classify(Xtrn, Ctrn, Xtst, kb)

print('Time taken to compute Knn = '+ str(time()-t) +' seconds')

# YourCode - Get confusion matrix and accuracy for each k in kb.

# YourCode - Save each confusion matrix.

cm1=my_confusion(Ctst,Cpreds[:,0])
scipy.io.savemat('cm1',{"cm1":cm1})

cm3=my_confusion(Ctst,Cpreds[:,1])
scipy.io.savemat('cm3',{"cm3":cm3})

cm5=my_confusion(Ctst,Cpreds[:,2])
scipy.io.savemat('cm5',{"cm5":cm5})

cm10=my_confusion(Ctst,Cpreds[:,3])
scipy.io.savemat('cm10',{"cm10":cm10})

cm20=my_confusion(Ctst,Cpreds[:,4])
scipy.io.savemat('cm20',{"cm20":cm20})

cms=[cm1, cm3, cm5, cm10, cm20]

# YourCode - Display the required information - k, N, Nerrs, acc for each element of kb

for i in range(len(kb)):
    print('K = ' + str(kb[i]) +' N = ' + str(len(Xtst)) + ' Nerrs = ' +  str(len(Cpreds) - np.trace(cms[i][0])) + ' Accuracy = '+ str(cms[i][1]))

## Generating the graphs below for different ks

# plt.plot([1,3,5,10,20],[cm1[1],cm3[1],cm5[1],cm10[1],cm20[1]],color='green', linestyle='dashed', linewidth = 3,
#          marker='o', markerfacecolor='blue', markersize=12)
# plt.axis([0,40,75,100])
# plt.ylim(70,100)
# plt.xlim(0,30)
# plt.title('Variation of Accuracy with K')
# plt.xlabel('Value of K')
# plt.ylabel('Accuracy ')
# plt.show()

