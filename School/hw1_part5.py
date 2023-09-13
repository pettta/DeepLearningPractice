import numpy as np
import pandas as pd 
from layers import *

#==== Data Setup====#
dataset = pd.read_csv("mcpd_augmented.csv")
X = dataset.iloc[0] # first observation, we could also feed it the whole dataset and take the first output at the end, either way this works 
print(X.shape)

#==== Layer Setup ====#
L1 = InputLayer(X)
L2 = FullyConnectedLayer(X.shape[0], 2) # since its just one row, the shape is just the number of columns
L3 = LogisticSigmoidLayer()
layers=[L1,L2,L3]

#==== Forward Pass ====#
h = X 
for layer in layers: 
    print("CLASS: ", layer.__class__.__name__)
    h = layer.forward(h)
    print("INPUT: ", layer.getPrevIn())
    print("OUTPUT: ", layer.getPrevOut())

