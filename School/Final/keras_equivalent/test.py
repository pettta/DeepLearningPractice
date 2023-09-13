import pandas as pd 
import numpy as np
from model import *
import matplotlib.pyplot as plt

# Plotting results
def plot_results(history1, history2):
    acc1 = history1.history['accuracy']
    val_acc1 = history1.history['val_accuracy']
    loss1 = history1.history['loss']
    val_loss1 = history1.history['val_loss']
    loss2 = history2.history['loss']
    val_loss2 = history2.history['val_loss']
    acc2 = history2.history['accuracy']
    val_acc2 = history2.history['val_accuracy']
    epochs = range(1, len(acc1) + 1)
    ## Accuracy plot
    plt.plot(epochs, acc1, 'b', label='Training acc')
    plt.plot(epochs, val_acc1, 'g', label='Validation acc')
    plt.plot(epochs, acc2, 'r', label='Training acc with CLR')
    plt.plot(epochs, val_acc2, 'y', label='Validation acc with CLR')
    plt.title('Training and validation accuracy')
    plt.legend()
    ## Loss plot
    plt.figure()
    plt.plot(epochs, loss1, 'b', label='Training loss')
    plt.plot(epochs, val_loss1, 'g', label='Validation loss')
    plt.plot(epochs, loss2, 'r', label='Training loss with CLR')
    plt.plot(epochs, val_loss2, 'y', label='Validation loss with CLR')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

# Read the data
df = pd.read_csv('../data/IMDB_Dataset.csv')
# shuffle
df = df.sample(frac=1).reset_index(drop=True)
train = df.iloc[:45000,:] # 90 percent of the data
test = df.iloc[45000:,:] # 10 percent of the data

# Hyperparameters for tokenizer
vocab_size = 10000
maxlen = 200 # only consider last 200 words of each review

# convert labels to binary 01 instead of pos/neg 1 = pos, 0 = neg
train['sentiment'] = train['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
test['sentiment'] = test['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# tokenizer 
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train['review'])
text_to_int_pad = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(train['review']), maxlen=maxlen, truncating='pre', padding='pre')
test_text_to_int_pad = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(test['review']), maxlen=maxlen, truncating='pre', padding='pre')

# split into train and validation
x_train = text_to_int_pad
y_train = train['sentiment'].values

x_test = test_text_to_int_pad
y_test = test['sentiment'].values

# Check Total Vocab Size
total_vocab_size = len(tokenizer.word_index) + 1
print('Total Vocabulary Size (Untrimmed): %d' % total_vocab_size)
print('Vocabulary Size (trimmed): %d' % vocab_size)

#--------- Define Classifier using Keras Sequential API---------#
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

model = keras.Sequential()
model.add(layers.Input(shape=(maxlen,)))
model.add(TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim))
model.add(TransformerBlock(embed_dim, num_heads, ff_dim))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(20, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1, activation="sigmoid"))

# check if using gpu
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
optimizer_name = "rmsprop" # NOTE: this is what I varied to test different optimizers
#NOTE: binary_crossentropy is typical log loss like we would expect from class
model.compile(optimizer=optimizer_name,
              loss="binary_crossentropy",
              metrics=["accuracy"])
clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular')
history1 = model.fit(x_train,
                    y_train,
                    batch_size=256,
                    epochs=4,
                    validation_data=(x_test, y_test), callbacks=[clr])  
history2 = model.fit(x_train,
                    y_train,
                    batch_size=256,
                    epochs=4,
                    validation_data=(x_test, y_test))
# plot results
plot_results(history1, history2)


