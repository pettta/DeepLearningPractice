from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.datasets import mnist 

# model creation 
model = keras.Sequential([
    layers.Dense(512, activation="relu"), 
    layers.Dense(10, activation="softmax")
])

# forward pass (optimizer finds gradients, loss is loss function ) 
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics =["accuracy"]) 

# data gathering 

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32") / 255

# model fitting (uses a batch generator for the images and labels epochs # of times and optimizes useing the previously set model.compile settings) 
model.fit(train_images, train_labels, epochs=5, batch_size=128) 

# model testing 
test_loss, test_acc = model.evaluate(test_images, test_labels) 
print(f"test_acc: {test_acc}") 
