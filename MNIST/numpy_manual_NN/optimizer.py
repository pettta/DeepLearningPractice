class SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    
    def apply_gradients(self, layer):
        layer.W += -self.learning_rate * layer.dweights 
        layer.b += -self.learning_rate * layer.dbiases 