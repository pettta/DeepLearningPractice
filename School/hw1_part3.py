import numpy as np
from layers import *
X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
b1 = np.array([1, 2])
w1 = np.array([[1, 2], [2, 0], [1, 1], [-1, 4]])
#Given input X
L1 = InputLayer(X)
L2 = FullyConnectedLayer(X.shape[1], b1.shape[0])
L2.setWeights(w1)
L2.setBiases(b1)
layers = [L1, L2, ReLuLayer(), LogisticSigmoidLayer(), SoftmaxLayer(), TanhLayer()]


# Input 
h = X 

#Activation layers

for layer in layers:
    connected_output = h.copy() 
    print("CLASS: ", layer.__class__.__name__)
    connected_output = layer.forward(connected_output)
    print("INPUT: ", layer.getPrevIn())
    print("OUTPUT: ", layer.getPrevOut())