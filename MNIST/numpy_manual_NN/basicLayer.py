import numpy as np 

class BasicDenseLayer: 
    def __init__(self, input_size, output_size, activation="relu"): 
        self.activation = activation
        self.W = 0.01 * np.random.randn(input_size, output_size)
        self.b = np.zeros((1, output_size))
    
    # Forward pass takes the input matrix from the previous layer and formats it into the next layer, with a proper activation function 
    def forwardPass(self, inputs):
        self.inputs = inputs 
        # Might seem a little redundant to have the matrix multiplication repeated, but no need to do it unless you have an existing activation function 
        if self.activation == "relu":
            inputLayer = np.matmul(inputs, self.W) + self.b
            self.output = np.maximum(0, inputLayer)
        elif self.activation == "tanh": 
            inputLayer = np.matmul(inputs, self.W) + self.b
            self.output = np.tanh(inputLayer)
        elif self.activation == "sigmoid": 
            inputLayer = np.matmul(inputs, self.W) + self.b
            self.output = BasicDenseLayer.sigmoid(inputLayer)
        elif self.activation == "softmax": 
            inputLayer = np.matmul(inputs, self.W) + self.b
            # This line forces the greatest value to be one and everything else less than it to prevent issues with large nums / overflow 
            # The axis is specified to get the sum of the rows rather than a single value
            expValues = np.exp(inputLayer - np.max(inputs, axis=1, keepdims=True))
            self.output = ( expValues / np.sum(expValues, axis=1, keepdims=True) )
        else:
            print("Unsupported activation function requested") 
    
    #Backwards pass for layers that also checks for the activation function gradients 
    def backwardsPass(self, dinputs, activation=""):
        if activation == "relu":
            dinputs[self.inputs <= 0] = 0 
        
        self.dweights = np.dot(self.inputs.T, dinputs)
        self.dbiases = np.sum(dinputs, axis=0, keepdims=True) 
        dinputs = np.dot(dinputs, self.W.T) 
        return dinputs
    
    
    
    # When called in the sequential, will take the proper inputs and run them in the model, then it spits out the output of that layer 
    def __call__(self, inputs):
        self.forwardPass(inputs)
        return self.output
      
    # Loss function for categorical crossentropy. Takes tensors of predicted and actual values 
    @staticmethod
    def calculateCategoricalCrossentropy(predictions, values):
        samples = len(predictions)
        # Clip the predictions that are zero to be extremely close to zero, so you don't get infinite error 
        predictionsClipped = np.clip(predictions, 1e-7, 1-1e-7)
        
        # Passed a scalar value 
        if len(values.shape) == 1: 
            correctConfidences = predictionsClipped[range(samples), values] 
           
        # One-hot encoded vector being passed 
        elif len(values.shape) == 2: 
            # everything but targets get multiplied by zero due to one-hot encoding 
            correctConfidences = np.sum(predictionsClipped*values, axis=1)
        
        # formula for CCE simplified due to nature of 1 being a successful identification 
        sampleLosses = -np.log(correctConfidences) 
        meanLoss = np.mean(sampleLosses) 
        return meanLoss
    
   
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
        
    @property 
    def weights(self):
        return [self.W, self.b]


class BasicSequential:
    def __init__(self, layers): 
        self.layers = layers

    # allows us to call the object like a function which then calls the layers like a function 
    def __call__(self, inputs):
        x = inputs
        for layer in self.layers: 
            x = layer(x)
        return x 

    # backpropogates from the softmax in order to get the changes in weights in biases (gradients)
    def backwardsPass(self, forwardOutput, trainingSolutions):
        # Start with the Crossentropy Loss backpropogation 
        samples = len(forwardOutput) 
        # Formatting check for one-hot encoding 
        if len(trainingSolutions.shape) == 2:
            trainingSolutions = np.argmax(trainingSolutions, axis=1)
        dinputs = forwardOutput.copy()
        # Gradient calculation and normalization
        dinputs[range(samples), trainingSolutions] -= 1 
        dinputs = dinputs/samples 
        
        # Dense layer 2 backpropogation 
        dinputs2 = self.layers[1].backwardsPass(dinputs, "") 
        
        # relu & dense layer 1 backpropogation
        dinputs1 = self.layers[0].backwardsPass(dinputs2, "relu") 
        

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights
        
