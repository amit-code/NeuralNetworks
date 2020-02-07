# Thanks to cs231n.github.io
#
#
# An example of using pixel-wise differences to compare two images with L1 distance (for one color channel in this example)
# Two images are subtracted elementwise and then all differences are added up to a single number.
# If two images are identical the result will be zero. But if the images are very different the result will be large.


# Use another function to load dataset to memory
from load_cifar_10 import  load_cifar_10_data

Xtr, train_filenames, Ytr, Xte,test_filenames, Yte,label_names =  load_cifar_10_data('cifar-10-batches-py')
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072