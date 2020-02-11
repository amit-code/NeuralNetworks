# Thanks to cs231n.github.io
#
#
# An example of using pixel-wise differences to compare two images with L1 distance (for one color channel in this example)
# Two images are subtracted elementwise and then all differences are added up to a single number.
# If two images are identical the result will be zero. But if the images are very different the result will be large.
# Compare the images pixel by pixel and add up all the differences.
# In other words, given two images and representing them as vectors I1,I2 ,
# a reasonable choice for comparing them might be the L1 distance

import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      #distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)  # L1 distance
      distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1)) # L2 distance
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred

# Use another function to load dataset to memory
from load_cifar_10 import  load_cifar_10_data

Xtr, train_filenames, Ytr, Xte,test_filenames, Yte,label_names =  load_cifar_10_data('cifar-10-batches-py')
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072


nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print('accuracy: %f' % ( np.mean(Yte_predict == Yte)))


# If you ran this code, you would see that this classifier only achieves 25% on CIFAR-10.
# That’s more impressive than guessing at random (which would give 10% accuracy since there are 10 classes)
# but nowhere near Convolutional Neural Networks that achieve about 95%

# The choice of distance. There are many other ways of computing distances between vectors
# Another common choice could be to instead use the L2 distance,
# which has the geometric interpretation of computing the euclidean distance between two vectors.
# In other words we would be computing the pixelwise difference as before,
# but this time we square all of them, add them up and finally take the square root.

# k-Nearest Neighbor Classifier. The idea is very simple: instead of finding the single closest image in the training set,
# we will find the top k closest images, and have them vote on the label of the test image.
# In particular, when k = 1, we recover the Nearest Neighbor classifier.

# The idea is to split our training set in two: a slightly smaller training set, and what we call a validation set
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
# Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
# Yval = Ytr[:1000]
# Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
# Ytr = Ytr[1000:]
# Cross-validation, idea is that instead of arbitrarily picking first 1000 datapoints to be validation set and rest training set


#########################################################################################################
# Linear Classification #
#########################################################################################################

# We are assuming that the image xi has all of its pixels flattened out to a single column vector of shape [D x 1]
# The matrix W (of size [K x D]), and the vector b (of size [K x 1]) are the parameters of the function
# K = 10, since there are 10 distinct classes (dog, cat, car, etc). D = 32 x 32 x 3 = 3072 pixels
#######################################################################
#  Loss function # Multiclass Support Vector Machine (Multiclass SVM)
#  f(xi,W,b)=Wxi+b
#
# The score function takes the pixels and computes the vector f(xi,W) of class scores, which we will abbreviate to s (short for scores)
# The score for the j-th class is the j-th element:   sj=f(xi,W)j
# The Multiclass SVM loss for the i-th example is then formalized as follows:
# Li=∑j≠yi max(0,sj−syi+Δ)
# The expression above sums over all incorrect classes (j≠yi)
# the SVM loss function wants the score of the correct class yi to be larger than the incorrect class scores by at least by Δ (delta).
# If this is not the case, we will accumulate loss. the threshold at zero max(0,−) function is often called the hinge loss
#########################################################################
# Regularization
#########################################################################
# The issue is that this set of W is not necessarily unique:
# there might be many similar W that correctly classify the examples
# we wish to encode some preference for a certain set of weights W over others to remove this ambiguity.
# We can do so by extending the loss function with a regularization penalty R(W)
# The most common regularization penalty is the L2, R(W)=∑k∑l W2 k,l
# L=1/N ∑i ∑j≠yi [max(0,f(xi;W)j − f(xi;W)yi + Δ)] + λ ∑k ∑l W2 k,l
########################################################################
#  Loss function # Softmax classifier
#######################################################################
# we now interpret these scores as the unnormalized log probabilities for each class and
# replace the hinge loss with a cross-entropy loss that has the form
#  Li=−log(efyi / ∑ j efj)
##########################################################################################
#   The loss function lets us quantify the quality of any particular set of weights W.
#   The goal of optimization is to find W that minimizes the loss function
#
#   The gradient is just a vector of slopes for each dimension in the input space
#   Now that we can compute the gradient of the loss function,
#   The procedure of repeatedly evaluating the gradient and then performing a parameter update is called Gradient Descent
#   The extreme case of this is a setting where the mini-batch contains only a single example.
#   This process is called Stochastic Gradient Descent (SGD)
#################################################################################
#  y = 2x , dy/dx =2, ...... y =2(x^2) ..  dy/dx = 2x, as the degree increases complexity increases for feature selection
#  for gradient decent target is to reduce partial derivates to ZERO, will return local minima or global minima. This will
#  direction of decent and values to be updated for weight to reach that decent

#########################################################################################

