import numpy as np 
import matplotlib.pyplot as plt
from layers import *

#==== Data Setup====#
w1 = 0.2 
learning_rates = [0.001, 0.01, 1.0, 5.0]

W1_history = []
J_history = []

all_W1_history = []
all_J_history = []



#=======For each example, do forward/backward prop, saving results=====#
for learning_rate in learning_rates:
    for epoch in range(100):       
        # x1 = 1, so it cancels out
        try:
            J = (1/4)*(w1)**4 - (4/3)*(w1)**3 + (3/2)*(w1)**2 
        except: # overflow error
            break
        dJdw1 = w1**3 - 4*w1**2 + 3*w1
        W1_history.append(w1)
        J_history.append(J)
        w1 -= learning_rate*dJdw1 
    all_W1_history.append(W1_history)
    all_J_history.append(J_history)
    W1_history = []
    J_history = []

#======Plot w vs J line plot for each====#
for loc, li in enumerate(all_W1_history):
    # plot epoch vs J
    plt.plot([i for i in range(len(all_J_history[loc]))], all_J_history[loc])
    plt.xlabel('epoch')
    plt.ylabel('J')
    print("FINAL VALUES: w1:", all_W1_history[loc][-1], "J:", all_J_history[loc][-1])
    if loc != 3:
        plt.plot([all_W1_history[loc][-1]], [all_J_history[loc][-1]], marker='*', color='r', ls='none')
    plt.ion()
    plt.show()
    plt.pause(1)
    plt.savefig(f"hw4_part4_fig{loc+1}.png")
    plt.close('all') 
    
print(all_W1_history, all_J_history)
