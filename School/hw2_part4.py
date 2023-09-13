import numpy as np
from layers import *

#==== Data Setup====#
Y1 = np.array([[0], [1]])
Y1_hat = np.array([[0.2], [0.3]])

Y2 = np.array([[1, 0, 0], [0, 1, 0]])
Y2_hat = np.array([[0.2, 0.2, 0.6], [0.2, 0.7, 0.1]])

#==== Objective Func Setup ====#
SE = SquaredError()
LL = LogLoss()
CE = CrossEntropy()


#==== Calculations====#
print("Squared Error: ")
print(SE.eval(Y1, Y1_hat))
print(SE.gradient(Y1, Y1_hat))

print("Log Loss: ")
print(LL.eval(Y1, Y1_hat))
print(LL.gradient(Y1, Y1_hat))

print("Cross Entropy: ")
print(CE.eval(Y2, Y2_hat))
print(CE.gradient(Y2, Y2_hat))