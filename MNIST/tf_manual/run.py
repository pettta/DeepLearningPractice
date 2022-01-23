from naive import NaiveDense, NaiveSequential
from batchGen import BatchGenerator 
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist 

# functions for the training loop 

def update_weights(gradients, weights):
    learning_rate = 1e-3 
    for g, w in zip(gradients, weights): 
        w.assign_sub(g * learning_rate) # -= the gradient times the learning rate of the cost function 
"""
Note: the above function would never be implemented like this practically. Instead you should use keras.optimizers 
and have the function run optimizer.apply_gradients(zip(gradients, weights)). But I wanna keep it like this, so I know
how it fundamentally works.

"""


def one_training_step(model, images_batch, labels_batch): 
    with tf.GradientTape() as tape: # model predictions done under GradientTape scope 
        predictions = model(images_batch) 
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(labels_batch, predictions) # calc loss values from actual
        average_loss = tf.reduce_mean(per_sample_losses)  
    gradients = tape.gradient(average_loss, model.weights) # finds direction of greatest increase in cost value that will affect the weights 
    update_weights(gradients, model.weights) # optimizes in right direction 
    return average_loss 


# full training loop using batch generator

def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches): 
            images_batch, labels_batch = batch_generator.next() 
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0: 
                print(f"loss at batch {batch_counter}: {loss:.2f}")


# create the model

model = NaiveSequential([
    NaiveDense(input_size=28*28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax) 
])


(train_images, train_labels), (test_images, test_labels) = mnist.load_data() 
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32") / 255 

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, epochs=10, batch_size=128) 

# Running the model 

predictions = model(test_images)
predictions = predictions.numpy() # conversion from tf tensor to numpy tensor
predicted_labels = np.argmax(predictions, axis=1) # sets equal to the indice of largest value ( what number it is most likely to be corresponds to an indice #) 
matches = predicted_labels == test_labels 
print(f"accuracy: {matches.mean():.2f}")




