import numpy as np 
import matplotlib.pyplot as plt
from layers import *

#==== Data Setup====#
W1 = [-1, 0.2, 0.9, 4]

W1_history = []
J_history = []

all_W1_history = []
all_J_history = []

learning_rate = 0.1

#=======For each example, do forward/backward prop, saving results=====#
for loc, w1 in enumerate(W1):
    for epoch in range(100):       
        # x1 = 1, so it cancels out
        J = (1/4)*(W1[loc])**4 - (4/3)*(W1[loc])**3 + (3/2)*(W1[loc])**2 
        dJdW1 = W1[loc]**3 - 4*W1[loc]**2 + 3*W1[loc]
        W1_history.append(W1[loc])
        J_history.append(J)
        W1[loc] -= learning_rate*dJdW1
    print(W1) 
    all_W1_history.append(W1_history)
    all_J_history.append(J_history)
    W1_history = []
    J_history = []
print(all_W1_history, all_J_history)

#======Plot w vs J line plot for each====#
for loc, li in enumerate(all_W1_history):
    # plot epoch vs J
    plt.plot([i for i in range(len(all_J_history[loc]))], all_J_history[loc])
    plt.xlabel('epoch')
    plt.ylabel('J')
    print("FINAL VALUES: w1:", all_W1_history[loc][-1], "J:", all_J_history[loc][-1])
    plt.plot([all_W1_history[loc][-1]], [all_J_history[loc][-1]], marker='*', color='r', ls='none')
    plt.ion()
    plt.show()
    plt.pause(1)
    plt.savefig(f"hw4_part3_fig{loc+1}.png")
    plt.close('all') 
    
