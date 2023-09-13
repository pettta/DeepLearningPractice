# NOTE: in case it was missed in the readme, I couldn't get this to work in time given my group size and other projects 
# I wanted to get some sort of results though, so I just implemented a version using keras and tensorflow 

from layers import * 
import numpy as np 
import pandas as pd 
from tensorflow import keras # NOTE: purely here for text word2vec purposes 

# Read the data
df = pd.read_csv('../data/IMDB_Dataset.csv')
# shuffle
df = df.sample(frac=1).reset_index(drop=True)
train = df.iloc[:45000,:] # 90 percent of the data
test = df.iloc[45000:,:] # 10 percent of the data

# Hyperparameters for tokenizer
vocab_size = 10000
maxlen = 200 # only consider last 200 words of each review

# Hyperparameters in general
lr = 1e-4 # TODO cyclical learning rate

# convert labels to binary 01 instead of pos/neg 1 = pos, 0 = neg
train['sentiment'] = train['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
test['sentiment'] = test['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train['review'])
text_to_int_pad = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(train['review']), maxlen=maxlen, truncating='pre', padding='pre')
test_text_to_int_pad = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(test['review']), maxlen=maxlen, truncating='pre', padding='pre')

# split into train and validation
x_train = text_to_int_pad
y_train = train['sentiment'].values
x_test = test_text_to_int_pad
y_test = test['sentiment'].values

print('x_train[0]:', x_train[0])

# Check Total Vocab Size
total_vocab_size = len(tokenizer.word_index) + 1
print('Total Vocabulary Size (Untrimmed): %d' % total_vocab_size)
print('Vocabulary Size (trimmed): %d' % vocab_size)

# Define Model Architecture as 
L1 = InputLayer(x_train)
L2 = EmbeddingLayer(x_train.shape[1], 100, vocab_size)
# transformer block begins
L3 = SelfAttentionLayer()
L4= LayerNorm()
    # FFN
L5 = FullyConnectedLayer()
L6 = ReLuLayer()
L7 = FullyConnectedLayer()
L8 = LayerNorm()
# transformer block ends
L9 = GlobalAveragePoolingLayer1D()
L9 = ReLuLayer()
L10 = SoftmaxLayer()
L11 = LogLoss()

layer_list = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11]

#=========== Learning ===========#
log_loss_train_list = []
log_loss_test_list = []
prev_log_loss_train = 0
prev_log_loss_test = 0
for i in range(100000):
    h = x_train.copy()
    h2 = x_test.copy()
    #==== Forward Pass ====#
    for loc, layer in enumerate(layer_list):
        if loc == len(layer_list) - 1:
            last_h_non_objective = h
            h = layer.eval(y_train, h)
            h2 = layer.eval(y_test, h2)
        else:
            h2 = layer.eval(h2, False) 
            h = layer.forward(h)

    current_log_loss_train = h
    current_log_loss_test = h2
    log_loss_train_list.append(current_log_loss_train)
    log_loss_test_list.append(current_log_loss_test)

    log_loss_test_diff = current_log_loss_test - prev_log_loss_test
    log_loss_train_diff = current_log_loss_train - prev_log_loss_train

    if i % 100 == 0:
        print('Iteration: %d, Log Loss Train: %.3f, Log Loss Test: %.3f' % (i, current_log_loss_train, current_log_loss_test))
    
    if (abs(log_loss_train_diff) <= 1e-8 and abs(log_loss_test_diff)!=0) and log_loss_test_diff > 0 :
        print("BROKE ON TRAINING LOSS DIFFERENCE", log_loss_train_diff)
        print("CONVERGED")
        break 

    #==== Backward Pass ====#
    grad = layer_list[-1].gradient(y_train, last_h_non_objective)
    grad = pd.DataFrame(grad)
    for layer in reversed(layer_list[:-1]):
        newGrad = layer.backward(grad)
        if(isinstance(layer, FullyConnectedLayer)):
            layer.updateWeights(grad, learningRate=lr)
        grad = newGrad
    prev_log_loss_train = current_log_loss_train
    prev_log_loss_test = current_log_loss_test


