from abc import ABC, abstractmethod 
import numpy as np 
class Layer(ABC):
    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []
    
    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn

    def setPrevOut(self, out): 
        self.__prevOut = out
    
    def getPrevIn(self):
        return self.__prevIn
    
    def getPrevOut(self):
        return self.__prevOut
    
    @abstractmethod
    def forward(self, dataIn): 
        pass

    @abstractmethod
    def gradient(self):
        pass 

    @abstractmethod
    def backward(self, gradIn): 
        pass 


#========= Input Layer ==========#
class InputLayer(Layer):
    #Input:  dataIn, an NxD matrix
    #Output:  None
    def __init__(self, dataIn):
        super().__init__() 
        self.__mean = np.mean(dataIn, axis=0)
        self.__std = np.std(dataIn, axis=0, ddof=1)
        

    #Input:  dataIn, an NxD matrix
    #Output: An NxD matrix
    def forward(self,dataIn, setValues=True):
        Y = (dataIn - self.__mean) / self.__std
        Y.fillna(0, inplace=True) # NOTE: fix for nan values
        if setValues:
            self.setPrevIn(dataIn)
            self.setPrevOut(Y)
        return Y

    # Method prototypes
    def gradient(self):
        pass

    def backward(self,gradIn):
        pass


#=========== Activation Layers ==========#
# NOTE for next assn, since I am using vector defn of gradients, make sure to use hadamard product, *
# NOTE except for the softmax layer, which is a special case, so make sure to use einsum for that case 

class LinearLayer(Layer):
    #Input:  None
    #Output:  None
    def __init__(self):
        super().__init__() 

    #Input:  dataIn, an NxK matrix
    #Output:  An NxK matrix
    def forward(self,dataIn,setValues=True):
        Y = dataIn
        if setValues:
            self.setPrevIn(dataIn)
            self.setPrevOut(Y)
        return Y 

    #Input: None
    #Output:  Identity tensor = 1 scalar (in effect)
    def gradient(self):
        return 1 

    def backward(self, gradIn):
        return gradIn * self.gradient()
  
class ReLuLayer(Layer):
    #Input:  None
    #Output:  None
    def __init__(self):
        super().__init__() 

    #Input:  dataIn, an NxK matrix
    #Output:  An NxK matrix
    def forward(self,dataIn, setValues=True):
        Y = np.maximum(0, dataIn)
        if setValues:
            self.setPrevIn(dataIn)
            self.setPrevOut(Y)
        return Y 

    #Input: None
    #Output:  Either an N by D matrix or an N by (D by D) tensor
    def gradient(self):
        return np.where(self.getPrevIn() >= 0, 1, 0)

    def backward(self, gradIn):
        return gradIn * self.gradient()
  

class LogisticSigmoidLayer(Layer):
    #Input:  None
    #Output:  None
    def __init__(self):
        super().__init__() 

    #Input:  dataIn, an NxK matrix
    #Output:  An NxK matrix
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        Y = 1/(1+np.exp(-dataIn))
        self.setPrevOut(Y)
        return Y
        

    #Input: None
    #Output:  Either an N by D matrix or an N by (D by D) tensor.
    def gradient(self):
        return self.getPrevOut() * (1 - self.getPrevOut())

    def backward(self, gradIn):
        return gradIn * self.gradient()
    


class SoftmaxLayer(Layer):
    #Input:  None
    #Output:  None
    def __init__(self):
        super().__init__() 

    #Input:  dataIn, an NxK matrix
    #Output:  An NxK matrix
    def forward(self,dataIn, setValues=True):
        # account for underflow 
        Y = np.exp(dataIn - np.max(dataIn)) / np.sum(np.exp(dataIn - np.max(dataIn)), axis=1, keepdims=True)
        if setValues:
            self.setPrevIn(dataIn)
            self.setPrevOut(Y)
        return Y 

    #Input: None
    #Output:  Either an N by D matrix or an N by (D by D) tensor
    def gradient(self):
        out = []
        for row in self.getPrevOut():
            out.append(np.diag(row) - (np.atleast_2d(row).T @ np.atleast_2d(row)))
        return np.stack(out)
    
    # Tensor product of gradIn and gradient
    def backward(self, gradIn):
        # TODO correct if wrong in later assn? seems right... 
        return np.einsum('ij, ijk->ik', gradIn, self.gradient())


class TanhLayer(Layer):
    #Input:  None
    #Output:  None
    def __init__(self):
        super().__init__()

    #Input:  dataIn, an NxK matrix
    #Output:  An NxK matrix
    def forward(self,dataIn, setValues=True):
         Y = np.tanh(dataIn) #NOTE I think this is okay? if not just use Y = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) + np.exp(-dataIn))
         if setValues:
            self.setPrevIn(dataIn)
            self.setPrevOut(Y)
         return Y 

    #Input: None
    #Output:  Either an N by D matrix or an N by (D by D) tensor
    def gradient(self):
        return 1 - self.getPrevOut()**2

    def backward(self, gradIn):
        return gradIn * self.gradient()


class LayerNorm(Layer):
    #Input:  None
    #Output:  None
    def __init__(self, gamma, epsilon=1e-8, beta=0):
        self.g = gamma
        self.e = epsilon
        self.b = beta
        super().__init__() 

    #Input:  dataIn, an NxK matrix
    #Output:  An NxK matrix
    def forward(self,dataIn,setValues=True):
        mean = np.mean(dataIn, axis=1, keepdims=True)
        variance = np.var(dataIn, axis=1, keepdims=True)
        x = (dataIn - mean) / np.sqrt(variance + self.e) # epsilon for numerical stability = 1e-8
        Y = self.g * x + self.b
        if setValues:
            self.setPrevIn(dataIn)
            self.setPrevOut(Y)
        return Y
    

    #Input: None
    #Output:  Identity tensor = 1 scalar (in effect)
    def gradient(self):
        return 1 

    def backward(self, gradIn):
        return gradIn * self.gradient()


#=========== Fully connected Layer ===========#

# NOTE: if we have to define many more update rules, we can make custom optimizer classes that have a updateWeights method, then this can call that and have an optimizer
class FullyConnectedLayer(Layer):
    #Input:  sizeIn, the number of features(columns) of data coming in
    #Input:  sizeOut, the number of features(columns) for the data coming out.
    #Output:  None
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        self.__weights = np.random.uniform(-1e-4, 1e-4, (sizeIn, sizeOut)) # random matrix of weights converts sizeIN to sizeOut
        self.__biases = np.random.uniform(-1e-4, 1e-4, (1, sizeOut)) # random vector of biases added to sizeOut
        self.__sw = np.zeros((sizeIn, sizeOut))
        self.__sb = np.zeros((1, sizeOut))
        self.__rw = np.zeros((sizeIn, sizeOut))
        self.__rb = np.zeros((1, sizeOut))

    #Input:  None
    #Output: The sizeIn x sizeOut weight matrix.
    def getWeights(self):
        return self.__weights

    #Input: The sizeIn x sizeOut weight matrix.
    #Output: None
    def setWeights(self, weights):
        self.__weights = weights

    #Input:  The 1 x sizeOut bias vector
    #Output: None
    def getBiases(self):
        return self.__biases

    #Input:  None
    #Output: The 1 x sizeOut biase vector
    def setBiases(self, biases):
        self.__biases = biases

    #====Set of methods for Adam optimizer====#
    def setSW(self, sw):
        self.__sw = sw
    
    def setSB(self, sb):
        self.__sb = sb
    
    def setRW(self, rw):
        self.__rw = rw

    def setRB(self, rb):
        self.__rb = rb
    #=========================================#

    #Input:  dataIn, an NxD data matrix
    #Output:  An NxK data matrix
    def forward(self,dataIn, setValues=True):
        #print("WEIGHTS", self.__weights) # TODO REMOVE
        #print("BIASES", self.__biases) # TODO REMOVE
        Y = np.dot(dataIn, self.__weights) + self.__biases
        #print("Fully connected Y", Y) # TODO REMOVE
        if setValues:
            self.setPrevIn(dataIn)
            self.setPrevOut(Y)
        return Y

    #Input: None
    #Output:  Either an N by D matrix or an N by (D by D) tensor   
    def gradient(self):
        return self.__weights.T

    def backward(self, gradIn):
        return gradIn @ self.gradient()
    
    def updateWeights(self, gradIn, learningRate, useAdam=False, rho1=0, rho2=0, t=0, useL2=False, l2alpha=0.01):
        dJdb = (np.sum(gradIn, axis=0) / gradIn.shape[0])[0] 
        if useL2:
            dJdW = ((1 - l2alpha) * (self.getPrevIn().T @ gradIn) / gradIn.shape[0] ) + (l2alpha * 2 * self.getWeights() / (self.getWeights().shape[0] * self.getWeights().shape[1]))
        else:
            dJdW = (self.getPrevIn().T @ gradIn) / gradIn.shape[0]
        if useAdam:
            self.setSW(self.__sw * rho1 + (1-rho1)*dJdW)
            self.setSB(self.__sb * rho1 + (1-rho1)*dJdb)
            self.setRW(self.__rw * rho2 + (1-rho2)*(dJdW * dJdW))
            self.setRB(self.__rb * rho2 + (1-rho2)*(dJdb * dJdb))
            sdenom= 1 - rho1**(t+1)
            rdenom = 1 - rho2**(t+1)
            if sdenom == 0:
                return 
            if rdenom == 0:
                return
            self.__weights -= learningRate * (self.__sw / (sdenom)) / (np.sqrt((self.__rw / (rdenom)))+1e-8)
            self.__biases -= learningRate * (self.__sb / (sdenom)) / (np.sqrt((self.__rb / (rdenom)))+1e-8)
        else:
            self.__weights -=  learningRate * dJdW
            self.__biases -= learningRate * dJdb

#========= Transformer Layers ===========#
class EmbeddingLayer(Layer):
    def __init__(self, max_position, featureSize, minFreq=1e-4):
        self.d=featureSize
        self.min_freq=minFreq
        self.min_freq=max_position
        super().__init__()
    
    def forward(self, dataIn, setValues=True):
        # Source from https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3 
        position = np.arange(self.max_position)
        freqs = self.min_freq**(2*(np.arange(self.d)//2)/self.d)
        pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
        pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
        pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
        if setValues:
            self.setPrevIn(dataIn)
            self.setPrevOut(pos_enc)
        return pos_enc
    
    def backward(self, gradIn):
        return gradIn
    
    def gradient(self):
        return 1


class SelfAttentionLayer(Layer):
    #Input:  sizeIn, the number of features(columns) of data coming in
    #Input:  sizeOut, the number of features(columns) for the data coming out.
    #Output:  None
    def __init__(self, sizeIn):
        super().__init__()
        self.Wkey = np.random.uniform(-1e-4, 1e-4, (sizeIn, sizeIn)) # random matrix of weights converts sizeIN to sizeOut
        self.Wquery = np.random.uniform(-1e-4, 1e-4, (sizeIn, sizeIn)) # random matrix of weights converts sizeIN to sizeOut
        self.Wvalue = np.random.uniform(-1e-4, 1e-4, (sizeIn, sizeIn)) # random matrix of weights converts sizeIN to sizeOut

    def forward(self, dataIn, setValues=True):
        K = np.dot(dataIn, self.Wkey)
        Q = np.dot(dataIn, self.Wquery)
        V = np.dot(dataIn, self.Wvalue)
        # do do product between each value of q and every value of k
        # then do softmax on each row of the resulting matrix
        # then do dot product between each row of the resulting matrix and each row of v
        # then concatenate the resulting matrix
        QK = np.einsum('ij, kj->ik', Q, K)
        QK = np.exp(QK - np.max(QK, axis=1, keepdims=True)) / np.sum(np.exp(QK - np.max(QK, axis=1, keepdims=True)), axis=1, keepdims=True)
        Y = np.dot(QK, V)
        if setValues:
            self.setPrevIn(dataIn)
            self.setPrevOut(Y)
        return Y

    #NOTE either hadamard multiply gradIn and gradient or do einsum ? 
    def backward(self, gradIn):
        pass
    
    def gradient(self):
        pass

#========= Objective Layers ===========#

class SquaredError():
  #Input: Y is an N by K matrix of target values.
  #Input: Yhat is an N by K matrix of estimated values.
  #Output:  A single floating point value.
  def eval(self,Y, Yhat):
    return np.mean((Y-Yhat)*(Y-Yhat))

  #Input: Y is an N by K matrix of target values.
  #Input: Yhat is an N by K matrix of estimated values.
  #Output:  An N by K matrix.
  def gradient(self,Y, Yhat):
    return -2 * (Y - Yhat) 

class LogLoss():
  #Input: Y is an N by K matrix of target values.
  #Input: Yhat is an N by K matrix of estimated values.
  #Output:  A single floating point value.
  def eval(self,Y, Yhat):
    return -1 * np.mean( (Y * np.log(Yhat)) + ((1-Y)*np.log(1-Yhat)) ) 

  #Input: Y is an N by K matrix of target values.
  #Input: Yhat is an N by K matrix of estimated values.
  #Output:  An N by K matrix.
  def gradient(self,Y, Yhat):
    return -1 * (Y - Yhat) / (Yhat * (1-Yhat)) 


class CrossEntropy():
  #Input: Y is an N by K matrix of target values.
  #Input: Yhat is an N by K matrix of estimated values.
  #Output:  A single floating point value.
  def eval(self,Y, Yhat):
    return -1 * np.mean(np.sum(Y * np.log(Yhat), axis=1))

  #Input: Y is an N by K matrix of target values.
  #Input: Yhat is an N by K matrix of estimated values.
  #Output:  An N by K matrix.
  def gradient(self,Y, Yhat):
    return -1 * (Y / Yhat)
  
#========= Special Layers ===========#
class DropoutLayer(Layer):
    #Input:  None
    #Output:  None
    def __init__(self, p):
        super().__init__()
        self.__p = p
        self.__mask = None # store mask of what stays and goes, easier way to do the multiplication

    #Input:  dataIn, an NxK matrix
    #Output:  An NxK matrix
    def forward(self,dataIn, setValues=True):
        if setValues:
            self.setPrevIn(dataIn)
            self.setPrevOut(dataIn)
            self.__mask = np.random.binomial(1, self.__p, size=dataIn.shape)
            return dataIn *(1/(1-self.__p)) *self.__mask
        else:
            return dataIn

    #Input: None
    #Output:  Either an N by D matrix or an N by (D by D) tensor
    def gradient(self):
        return (1/1-self.__p) * (self.__mask != 0)

    def backward(self, gradIn):
        return gradIn * self.gradient()


class GlobalAveragePoolingLayer1D(Layer):
    #Input:  None
    #Output:  None
    def __init__(self):
        super().__init__()

    #Input:  dataIn, an NxK matrix
    #Output:  An NxK matrix
    def forward(self,dataIn, setValues=True):
        Y = np.mean(dataIn, axis=1)
        if setValues:
            self.setPrevIn(dataIn)
            self.setPrevOut(Y)
        return Y

    #Input: None
    #Output:  Either an N by D matrix or an N by (D by D) tensor
    def gradient(self):
        return 1

    def backward(self, gradIn):
        return gradIn * self.gradient()