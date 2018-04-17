# Machine-Learning-Classification-
Classifier for EMNIST data set using KNN,  Bernoulli naive Bayes classification and Bayes classification with Gaussian distributions

There are 1800 training samples and 300 test samples for each class. 

The data set contains the following arrays/vectors:

Name Size               (Class)           Description

dataset.train.images  46800x784 (uint8)   training samples

dataset.train.labels  46800x1 (double)    class labels of training samples

dataset.test.images   7800x784 (uint8)    test samples

dataset.test.labels   7800x1 (double)     class labels of test samples

Each pixel value is represented as an unsigned byte integer (uint8) with the range in [0, 255]. 

A class label is represented as an integer number between 1 and 26, where 1 denotes ’A’ and 26
’Z’, respectively. The data set is supposed to contain letters in either upper case or lower case only,
but you should expect that the actual data set allocated to you may contain both.
