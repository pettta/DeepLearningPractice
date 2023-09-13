import numpy as np 
import matplotlib.pyplot as plt
from layers import *
w_list = []
J_list = []

#======= Iterate over w values, generating J values =====#
for w in np.arange(-2, 5, 0.1):
    # x1 = 1, so it cancels out
    J = (1/4)*(w)**4 - (4/3)*(w)**3 + (3/2)*(w)**2 
    w_list.append(w)
    J_list.append(J)

#======Plot w vs J line plot====#
plt.plot(w_list, J_list)
plt.xlabel('w')
plt.ylabel('J')
plt.show()
