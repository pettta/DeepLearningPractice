import numpy as np 
import matplotlib.pyplot as plt
import time
from layers import *

#==== Data Setup====#
w1 = 0.2 
learning_rate = 5
rho1 = 0.9
rho2 = 0.999
delta = 1e-8

s=0
r=0

W1_history = []
J_history = []


#=======For each example, do forward/backward prop, saving results=====#
for epoch in range(100):   
    print("s:", s, "r:", r)
    time.sleep(0.1)
    # x1 = 1, so it cancels out
    J = (1/4)*(w1)**4 - (4/3)*(w1)**3 + (3/2)*(w1)**2 
    dJdw1 = w1**3 - 4*w1**2 + 3*w1

    # will have to change a little bit in next part due to shape mattering rather than dealing with scalars
    s = rho1*s + (1-rho1)*dJdw1 
    r = rho2*r + (1-rho2)*(dJdw1 * dJdw1)

    W1_history.append(w1)
    J_history.append(J)

    sdenom = 1 - (rho1**epoch)
    rdenom = 1 - (rho2**epoch)

    if sdenom == 0:
        continue
    if rdenom == 0:
        continue

    w1 -= learning_rate * ((s/(sdenom)) / (np.sqrt((r / (rdenom)))+delta))


#======Plot epoch vs J line plot====#
plt.plot([i for i in range(len(J_history))], J_history)
plt.xlabel('epoch')
plt.ylabel('J')
print("FINAL VALUES: w1:", W1_history[-1], "J:", J_history[-1])
plt.ion()
plt.show()
plt.savefig(f"hw4_part5_adam_plot.png")
plt.pause(2)
plt.close('all')

