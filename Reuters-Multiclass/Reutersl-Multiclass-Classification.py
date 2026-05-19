import keras
from keras.datasets import reuters
from keras import layers
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
	num_words=10000
)


def multi_hot_encode(sequences, num_classes):
	#this creates a zeros matrix with 25000 rows and 10000 columns
	results = np.zeros((len(sequences), num_classes))
	#this analyzes each row, and puts a 1 on where each word is indexed.             Enumerate gives us both the row index and his content.
	for i, sequence in enumerate(sequences):
		results[i][sequence] = 1.0
	return results

x_train = multi_hot_encode(train_data, num_classes=10000)
x_test = multi_hot_encode(test_data, num_classes=10000)


y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)


model = keras.Sequential(
	[
		layers.Dense(64, activation="relu"),
		layers.Dense(64, activation="relu"),
		layers.Dense(46, activation="softmax"),
	]
)

top_3_accuracy = keras.metrics.TopKCategoricalAccuracy(
k=3, name="top_3_accuracy"
)

model.compile(
	optimizer="adam",
	loss="categorical_crossentropy",
	metrics=["accuracy", top_3_accuracy],
)

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]


history = model.fit(
	partial_x_train,
	partial_y_train,
	epochs=20,
	batch_size=512,
	validation_data=(x_val, y_val),
)

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Loss")
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)