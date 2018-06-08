from __future__ import print_function
import math
import keras
import numpy
from keras import backend as K

print("math.exp(4):", math.exp(4))
print(numpy.random.rand(3,2))
#K_square = K.square(numpy.random.rand(3,2))
K_square = K.print_tensor(K.square, message = "keras.square() is ")
K_square = K.square(numpy.random.rand(3,2))

#print("keras.square():", keras.square(numpy.random.rand(3,2)))
