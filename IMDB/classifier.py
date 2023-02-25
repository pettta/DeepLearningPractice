"""
Basic not super accurate IMDB review pos/neg binary classifcation
"""
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.datasets import imdb 
import numpy as np 
import matplotlib.pyplot as plt 

def decode(source, review_num):
    word_index = imdb.get_word_index() 
    # int: word instead of word: int 
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = " ".join([reverse_word_index.get(i-3, "?") for i in source[review_num]]) 
    return decoded_review 

def vectorize_sequences(sequences, ndim=10000): 
    # len(sequences) x ndim matrice 
    results=np.zeros((len(sequences), ndim)) 
    for loc, sequence in enumerate(sequences): 
        for val in sequence: 
            # Sets specific indicies of results[loc] matrice to 1s
            results[loc, val] = 1 
    return results 

# Model Definition 
"""
matrix W will have shape input_ndims x 16 that relu gets applied to
lower dimensional representational space (16) here reflects low complexity classification
You need a non-linear transformation/activation function to the input matrice to avoid being stuck in the 16dim space
Since this is a bin classification problem, we'll use binary_crossentropy loss and last layer sigmoid activiation
"""
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
]) 

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# limit to 10,000 most occuring words --> if unlimited many words only occur once
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) 

# Vectorize the training and test data 
x_train = vectorize_sequences(train_data) 
x_test = vectorize_sequences(test_data) 

# Vectorize labels 
y_train = np.asarray(train_labels).astype("float32") 
y_test = np.asarray(test_labels).astype("float32")

# Compare vectorized vs non-vectorized training data && labels 
"""
print(train_data[0])
print(x_train[0])
print(train_labels[0]) 
print(y_train[0])
"""

# Set apart train and validation data 
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Train the model 
history = model.fit(partial_x_train, partial_y_train, epochs=5, batch_size=512, validation_data=(x_val, y_val))

# Displaying training and validation loss && accuracy 
"""
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss") 
plt.title("Training and validation loss") 
plt.xlabel("Epochs") 
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
"""

# Use the model for predictions 
print(model.predict(x_test)) 



