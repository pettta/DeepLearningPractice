import tensorflow as tf 

class NaiveDense: 
    def __init__(self, input_size, output_size, activation): 
        self.activation = activation
           
        w_shape = (input_size, output_size) 
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value) 
                
        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b) 
        

    @property 
    def weights(self):
        return [self.W, self.b]


class NaiveSequential:
    def __init__(self, layers): 
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers: 
            x = layer(x)
        return x 

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights
