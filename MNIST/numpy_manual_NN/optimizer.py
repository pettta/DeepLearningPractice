import numpy as np 
class SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    
    def apply_gradients(self, layer): 
        #print("WEIGHTS BEFORE: ", layer.W) 
        layer.W = layer.W - (self.learning_rate * layer.dweights)
        #print("WEIGHTS AFTER: ", layer.W) 
        #print("BIASES BEFORE: ", layer.b) 
        layer.b = layer.b - (self.learning_rate * layer.dbiases) 
        #print("BIASES AFTER: ", layer.b) 
        