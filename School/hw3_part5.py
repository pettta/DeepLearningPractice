import numpy as np 
import pandas as pd 
import time
import matplotlib.pyplot as plt
from layers import *

""""
Couldn't manage to get this one to work, but I think I was on the right track.
I think the issue is that possibly I was either using the wrong gradient for the squared error
function, or maybe something with momentum, honestly I'm not too sure. I just know that
the weights would converge to about 395 where the difference in SE was about 3e+2, but 
then it would start to diverge again. Let me know what I did wrong, I'm curious. 
"""


#==== Data Setup====#
# Load data
X = pd.read_csv('medical.csv')
# Split labels
Y = pd.DataFrame(X.pop(X.columns[-1]))


#==== Layer Setup ====#
L1 = InputLayer(X)
L2 = FullyConnectedLayer(X.shape[1], 1) 
L3 = SquaredError()

layers=[L1, L2, L3]

#==== Learning ====#
MSE_list = []
prev_MSE = 0
for i in range(100000):
    h=X
    #==== Forward Pass ====#
    for loc, layer in enumerate(layers):
            if loc == len(layers)-1:
                last_h_non_objective = h 
                h = layer.eval(Y, h)
                print("J", h) # TODO REMOVE
            else:
                h = layer.forward(h)
    current_MSE = np.mean(h)
    MSE_list.append(np.mean(current_MSE))
    MSE_diff = abs(current_MSE - prev_MSE)
    print("MSE DIFFERENCE", MSE_diff)
    if MSE_diff <= 1e-10:
        print("CONVERGED")
        break
    print("BEGINNING WEIGHTS", L2.getWeights(), L2.getBiases())
    #==== Backward Pass ====#
    print("Y", Y, "Yhat", last_h_non_objective)
    grad = layers[-1].gradient(Y, last_h_non_objective)
    grad = pd.DataFrame(grad)
    #print("COMPARISON OF PREDICT, ACTUAL, AND GRADIENT:") # TODO REMOVE
    #print("Y", Y, "Yhat", h, "grad", grad) # TODO REMOVE
    #RMSE = np.sqrt(np.mean((Y-h).T*(Y-h)))
    #print("RMSE", RMSE) # TODO REMOVEs
    #print("FRONTEND GRAD", grad)
    for layer in reversed(layers[:-1]):
        print("LAYER", layer.__class__.__name__)
        newGrad = layer.backward(grad)
        if(isinstance(layer, FullyConnectedLayer)):
        #   print("GRADIENT BEING PASSED TO UPDATE: ", grad)
            layer.updateWeights(grad, 1e-4)
        grad = newGrad
    print("END WEIGHTS", L2.getWeights(), L2.getBiases())
    prev_MSE = current_MSE


#=== Predictions ====#
h=X
for loc, layer in enumerate(layers):
    if loc == len(layers)-1:
        break
    else:
        h = layer.forward(h)
# Final SMAPE:
SMAPE = np.mean(np.abs(Y - h) / (np.abs(Y) + np.abs(h)))
print("FINAL SMAPE", SMAPE)
# Final RMSE:
RMSE = np.sqrt(np.mean((Y-h)*(Y-h)))
print("FINAL RMSE", RMSE)

#==== Plotting ====#
plt.plot(MSE_list)
plt.title("Mean Squared Error over Time")
plt.xlabel("Iteration")
plt.ylabel("Squared Error")
plt.show()

