import numpy as np 
import pandas as pd 
import time
import matplotlib.pyplot as plt
from layers import *


#==== Data Setup====#
# Load data
X = pd.read_csv('KidCreative.csv')
# Split labels
Y = pd.DataFrame(X.pop(X.columns[1]))

#==== Layer Setup ====#
L1 = InputLayer(X)
L2 = FullyConnectedLayer(X.shape[1], 1) 
L3 = LogisticSigmoidLayer() 
L4 = LogLoss()
layers=[L1, L2, L3, L4]

#==== Learning ====#
log_loss_list = []
prev_log_loss = 0 
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
    current_log_loss = h
    log_loss_list.append(current_log_loss)
    log_loss_diff = abs(current_log_loss - prev_log_loss)
    print("LOG LOSS DIFFERENCE", log_loss_diff)
    if log_loss_diff <= 1e-10:
        print("CONVERGED")
        break 
    print("BEGINNING WEIGHTS", L2.getWeights(), L2.getBiases())
    #==== Backward Pass ====#
    grad = layers[-1].gradient(Y, last_h_non_objective)
    grad = pd.DataFrame(grad)
    for layer in reversed(layers[:-1]):
        newGrad = layer.backward(grad)
        if(isinstance(layer, FullyConnectedLayer)):
            layer.updateWeights(grad, 1e-4)
        grad = newGrad
    prev_log_loss = current_log_loss

#=== Predictions ====#
h=X
for loc, layer in enumerate(layers):
    if loc == len(layers)-1:
        break
    else:
        h = layer.forward(h)
# 0.5 theshold for classification 
h = pd.DataFrame(h) # allows for matlab style filtering 
h[h >= 0.5] = 1
h[h < 0.5] = 0
print("FINAL PREDICTION", h)
print("FINAL ACTUAL", Y)
report_df = pd.DataFrame({'y':Y.iloc[:,0], 'y_pred':h.iloc[:,0]})
acc = len(report_df.loc[report_df["y"] == report_df["y_pred"]].index) / len(report_df.index)
print("FINAL TRAINING ACCURACY", acc)

#==== Plotting ====#
#Plot log loss over time
plt.plot(log_loss_list)
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("Log Loss over Time")
plt.show()
