import numpy as np
from layers import *

#==== Data Setup====#
X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
b1 = np.array([1, 2])
w1 = np.array([[1, 2], [2, 0], [1, 1], [-1, 4]])

#==== Layer Setup ====#
L1 = InputLayer(X)
L2 = FullyConnectedLayer(X.shape[1], b1.shape[0])
L2.setWeights(w1)
L2.setBiases(b1)
L3 = LogisticSigmoidLayer()
layers=[L1,L2,L3]

#==== Forward Pass ====#
h = X 
for layer in layers: 
    print("CLASS: ", layer.__class__.__name__)
    h = layer.forward(h)
    print("INPUT: ", layer.getPrevIn())
    print("OUTPUT: ", layer.getPrevOut())
