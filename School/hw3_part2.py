import numpy as np 
import matplotlib.pyplot as plt
from layers import *

#==== Data Setup====#
W = [0, 0]
W1_history = []
W2_history = []
J_history = []
learning_rate = 0.01

for i in range(100):
    J = (W[0] - 5*W[1] -2 )**2
    dJdW1 = 2*(W[0] - 5*W[1] -2)
    dJdW2 = -10*(W[0] - 5*W[1] -2)
    W1_history.append(W[0])
    W2_history.append(W[1])
    print(W[0], W[1], J)
    J_history.append(J)
    W[0] = W[0] - learning_rate*dJdW1
    W[1] = W[1] - learning_rate*dJdW2

# Plot w[0] vs w[1] vs J line plot 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(W1_history, W2_history, J_history, c='r', marker='o')
# Set labels for the axes (optional)
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('J')
plt.show()