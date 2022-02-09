from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.datasets import mnist

#make a convnet for the mnist data that starts on a 28x28x1 tensor
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu") (inputs) 
x = layers.MaxPooling2D(pool_size=2) (x) 
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2) (x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x) # height and width decrease & channels increase with each successive layer (here = 3x3x128)
x = layers.Flatten() (x) # flattened rank 3 tensor to vector to be processed by Dense layer 

#choice of 10 numbers, softmax for if it is or not a digit
outputs = layers.Dense(10, activation="softmax")(x)

#create the model
model = keras.Model(inputs=inputs, outputs=outputs)

#data initialization
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255 
test_images = test_images.reshape((10000, 28, 28, 1 )) 
test_images = test_images.astype("float32") / 255

#run the model
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5, batch_size=32, verbose=2) 

#evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.3f}")

