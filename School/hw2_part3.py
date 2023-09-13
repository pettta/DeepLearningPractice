import numpy as np
from layers import *

#==== Data Setup====#
H = np.array([[1, 2, 3], [4, 5, 6]])
b1 = np.array([-1, 2])
w1 = np.array([[1, 2], [3, 4], [5, 6]])

#==== Layer Setup ====#
L2 = FullyConnectedLayer(H.shape[1], b1.shape[0])
L2.setWeights(w1)
L2.setBiases(b1)
layers = [L2, ReLuLayer(), LogisticSigmoidLayer(), SoftmaxLayer(), TanhLayer()]


#==== Forward Pass  Calling Gradients====#
h = H
for layer in layers: 
    connected_output = h.copy() 
    print("CLASS: ", layer.__class__.__name__)
    connected_output = layer.forward(connected_output)
    print("INPUT: ", layer.getPrevIn())
    print("OUTPUT: ", layer.getPrevOut())
    print("GRADIENT: ", layer.gradient())