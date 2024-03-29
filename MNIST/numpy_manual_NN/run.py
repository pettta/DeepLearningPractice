from basicLayer import BasicDenseLayer, BasicSequential
from optimizer import SGD 
import numpy as np
from batchGen import BatchGenerator 
from tensorflow.keras.datasets import mnist 

optimizer = SGD(learning_rate=1e-3) 

def update_weights(layers): 
    #print("BEFORE WEIGHTS0:", layers[0].W, "\nBIASES0:", layers[0].b) 
    optimizer.apply_gradients(layers[0])         
    #print("AFTER WEIGHTS0:", layers[0].W, "\nBIASES0:", layers[0].b) 
    #print("BEFORE WEIGHTS1:", layers[1].W, "\nBIASES1:", layers[1].b)
    optimizer.apply_gradients(layers[1])      
    #print("AFTER WEIGHTS1:", layers[1].W, "\nBIASES1:", layers[1].b)


def one_training_step(model, images_batch, labels_batch): 
    predictions = model(images_batch) 
    average_loss = BasicDenseLayer.calculateCategoricalCrossentropy(predictions, labels_batch) # calc loss values from actual 
    # Backpropagation breakdown into parts 
    model.backwardsPass(predictions, labels_batch) 
    # update weights and biases of each of the layers given the changes calculated prior
    update_weights(model.layers) # optimizes in right direction 
    return average_loss 

# full training loop using batch generator

def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches): 
            images_batch, labels_batch = batch_generator.next() 
            loss = one_training_step(model, images_batch, labels_batch) # would fail here, go to function an implement 
            if batch_counter % 100 == 0: 
                print(f"loss at batch {batch_counter}: loss {loss:.2f}")

# Data set up 

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() 
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32") / 255 

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32") / 255

# Model Creation 

model = BasicSequential([
    BasicDenseLayer(input_size=28*28, output_size=512, activation="relu"),
    BasicDenseLayer(input_size=512, output_size=10, activation="softmax") 
])


# Forward pass & run 
fit(model, train_images, train_labels, epochs=20, batch_size=128)


# Testing accuracy

predictions = model(test_images)
predicted_labels = np.argmax(predictions, axis=1) # sets equal to the indice of largest value ( what number it is most likely to be corresponds to an indice #) 
matches = predicted_labels == test_labels 
print(f"accuracy: {matches.mean():.2f}")
