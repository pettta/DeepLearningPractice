from tensorflow import keras 
from tensorflow.keras.datasets import reuters
from tensorflow.keras import layers 
from tensorflow.keras.utils import to_categorical
import numpy as np 
import matplotlib.pyplot as plt 
 

def decode(source, review_num):
    word_index = reuters.get_word_index() 
    # int: word instead of word: int 
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_newswire = " ".join([reverse_word_index.get(i-3, "?") for i in source[review_num]]) 
    return decoded_newswire

def vectorize_sequences(sequences, ndim=10000): 
    # len(sequences) x ndim matrice 
    results=np.zeros((len(sequences), ndim)) 
    for loc, sequence in enumerate(sequences): 
        for val in sequence: 
            # Sets specific indicies of results[loc] matrice to 1s
            results[loc, val] = 1 
    return results 

# Note this is basically the functionality that we are getting out of the keras function to_categorical 
"""
def to_one_hot(labels, ndim=46):
    results = np.zeros((len(labels), ndim)) 
    for loc, label in enumerate(labels):
        results[i, label] = 1
    return results 
"""

# Model Definition 
"""
Larger layers as to not lose relevant information for learning 
46 size softmax for getting prob dist of what category it fits into 
"""
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"), 
    layers.Dense(46, activation="softmax")
])

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]) 

# Vectorize and 1-hot training and testing labels / data 
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data) 
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels) 


# Set apart train and validation data 
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# Train the model 
history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))

# Plotting the training and validation loss 
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plotting hte training and validation accuracy
plt.clf()
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()