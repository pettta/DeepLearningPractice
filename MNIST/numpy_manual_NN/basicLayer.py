import numpy as np 
import random 

class BasicDenseLayer: 
    def __init__(self, input_size, output_size, activation="relu"): 
        self.activation = activation
        self.W = 0.01 * np.random.randn(input_size, output_size)
        self.b = np.zeros((1, output_size))
    
    # Forward pass takes the input matrix from the previous layer and formats it into the next layer, with a proper activation function 
    def forwardPass(self, inputs):
        self.inputs = inputs.copy() 
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
    """ 
    Gradient Calculation explanation:
        Let zL = the output of matmul(input, w) + b at layer L 
        Let aL = the activated function, ie RELU(zL) = aL 
        We want to find both: dC/dwL and dC/dbL: 
            dC/dwL = dC/daL * daL/dzL * dzL/dwL  
            dC/dbL = dC/daL * daL/dzL * dzL/dbL
                dC/daL = dinputs at the beginning of the function 
                daL/dzL = dRelu --> the change that dRelu makes on dinputs in this case 
                dzL/dwL = aL-1 = inputs 
                dzL/dbL = 1 
                
    Therefore for a single neuron on layer L: 
            dC/dwL = dinputs (after relu derivative) * inputs
            dC/dbL = dinputs (after relu derivative) 
    We can then backpropogate further getting on layer L-1:
            dC/daL-1 = wL * dinputs 
    """
    def backwardsPass(self, dinputs, activation=""):
        dvalues = dinputs.copy() 
        if activation == "relu":
            dvalues = BasicDenseLayer.deltaRelu(dvalues)
            
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) 
        dvalues = np.dot(dvalues, self.W.T) 
        return dvalues
    
    
    
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
        
    @staticmethod
    def deltaRelu(x):
        x[x<0] = 0
        return x 
        
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
        # Start with the Crossentropy Loss & Softmax backpropogation 
        samples = trainingSolutions.shape[0]
        # Need to do another softmax of the softmax layer output 
        outputArray = forwardOutput.copy() 
        exps = np.exp(outputArray)
        dinputs = exps/ np.sum(exps)
        # Gradient calculation and normalization
        dinputs[range(samples), trainingSolutions] -= 1 
        # Average the gradient across all training samples 
        dinputs = dinputs/samples 
        
        # Dense layer 2 backpropogation using previous gradient 
        dinputs2 = self.layers[1].backwardsPass(dinputs, "") 
        
        # relu & dense layer 1 backpropogation
        dinputs1 = self.layers[0].backwardsPass(dinputs2, "relu") 
        

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights
        
