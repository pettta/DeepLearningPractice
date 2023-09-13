import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from layers import *


#==== Data Setup====#
# Load data
X_train = pd.read_csv('mnist_train_100.csv', header=None)
X_test = pd.read_csv('mnist_valid_10.csv', header=None) 

# Shuffle data (including labels)
# NOTE: if you want to ensure reproducibility, set random_state=0
X_train = X_train.sample(frac=1)
X_test = X_test.sample(frac=1)

# Split labels
Y_train = pd.DataFrame(X_train.pop(X_train.columns[0]))
Y_test = pd.DataFrame(X_test.pop(X_test.columns[0]))

#One hot encode labels: for cross entropy objective function
Y_train = pd.get_dummies(Y_train.iloc[:,0], dtype=float)
Y_test = pd.get_dummies(Y_test.iloc[:,0], dtype=float)

"""
# Set up ADAM hyperparameters
rho1 = 0.9
rho2 = 0.999"""
lr = 1e-2

#==== Layer Setup ====#
L1 = InputLayer(X_train)
L2 = FullyConnectedLayer(X_train.shape[1], 100)
L3 = ReLuLayer()
L4 = FullyConnectedLayer(100, 10)
L5 = SoftmaxLayer()
L6 = CrossEntropy()
layers=[L1, L2, L3, L4, L5, L6]

#==== Learning ====#
centropy_train_loss_list = []
centropy_test_loss_list = []
prev_train_centropy_loss = 0 
prev_test_centropy_loss = 0

for i in range(10000):
    h=X_train.copy()
    h2=X_test.copy()
    #==== Forward Pass====#
    for loc, layer in enumerate(layers):
            #print("LAYER", layer.__class__.__name__)
            #print("H", h)
            if loc == len(layers)-1:
                last_h_non_objective = h 
                h = layer.eval(Y_train, h)
                h2 = layer.eval(Y_test, h2)
            else:
                h2 = layer.forward(h2, False) # we don't want it to adjust the last input and ouput for the test set
                # deal with infinities and nans
                h2[h2 == -np.inf] = 0
                h2[h2 == np.inf] = 0
                h2 = np.nan_to_num(h2)
                h = layer.forward(h)
                
    
    current_centropy_train_loss = h
    current_centropy_test_loss = h2
    centropy_train_loss_list.append(current_centropy_train_loss)
    centropy_test_loss_list.append(current_centropy_test_loss)
    
    #NOTE: possibly use gap between train and test loss as a stopping criterion
    centropy_test_loss_diff = current_centropy_test_loss - prev_test_centropy_loss
    centropy_train_loss_diff = current_centropy_train_loss - prev_train_centropy_loss
    if i % 100 == 0: # print only every 100 iterations
        print("CROSS ENTROPY TESTING LOSS DIFFERENCE", centropy_test_loss_diff)
    
    if centropy_train_loss_diff <= 1e-7 and centropy_test_loss_diff > 0:
        print("CONVERGED")
        break 
    
    #==== Backward Pass ====#
    grad = layers[-1].gradient(Y_train, last_h_non_objective)
    grad = pd.DataFrame(grad)
    for layer in reversed(layers[:-1]):
        newGrad = layer.backward(grad)
        if(isinstance(layer, FullyConnectedLayer)):
            layer.updateWeights(grad, learningRate=lr)
        grad = newGrad
    prev_train_centropy_loss = current_centropy_train_loss
    prev_test_centropy_loss = current_centropy_test_loss



#=== Predictions (validation) ====#
h3=X_test
h4=X_train
for loc, layer in enumerate(layers):
    print("LAYER", layer.__class__.__name__)
    if loc == len(layers)-1:
        break
    else:
        h3 = layer.forward(h3, False)
        h3[h3 == -np.inf] = 0
        h3[h3 == np.inf] = 0
        h3 = np.nan_to_num(h3)
        h4 = layer.forward(h4, False)
# use argmax to select the class with the highest probability 
h3 = pd.DataFrame(h3) # allows for matlab style filtering
h4 = pd.DataFrame(h4) # allows for matlab style filtering 
h3 = h3.idxmax(axis=1).reset_index(drop=True) # somewhere along the way row indices got messed up, but should still be right
h4 = h4.idxmax(axis=1).reset_index(drop=True)
Y_test = Y_test.idxmax(axis=1).reset_index(drop=True)
Y_train = Y_train.idxmax(axis=1).reset_index(drop=True)
print("FINAL PREDICTION", h3)
print("FINAL ACTUAL", Y_test)
test_report_df = pd.DataFrame({'y':Y_test, 'y_pred':h3})
train_report_df = pd.DataFrame({'y':Y_train, 'y_pred':h4})
print("REPORT DF", test_report_df)
test_acc = np.sum(h3 == Y_test) / len(Y_test)
train_acc = np.sum(h4 == Y_train) / len(Y_train)
print("FINAL TEST ACCURACY", test_acc)
print("FINAL TRAIN ACCURACY", train_acc)


#==== Plotting ====#
plt.plot(centropy_train_loss_list, label="Train Loss")
plt.plot(centropy_test_loss_list, label="Test Loss")
plt.xlabel("Iteration")
plt.ylabel("Cross Entropy Loss")
plt.title("Cross Entropy Loss over Time")
plt.legend()
plt.ion()
plt.show()
plt.pause(2) 
plt.savefig(f"hw5_part2_cross_entropy_plot.png")
