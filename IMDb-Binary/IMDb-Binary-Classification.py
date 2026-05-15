import keras
from keras.datasets import imdb
import numpy as np
from keras import layers
import matplotlib.pyplot as plt


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
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


#labels are already arrays, we just need to convert their type
y_train = train_labels.astype("float32")
y_test = test_labels.astype("float32")

model = keras.Sequential(
	[
		layers.Dense(16, activation="relu"),
		layers.Dense(16, activation="relu"),
		layers.Dense(1, activation="sigmoid"),
	]
)

model.compile(
	optimizer="adam",
	loss="binary_crossentropy",
	metrics=["accuracy"],
)


history = model.fit(
	x_train,
	y_train,
	epochs=4,
	batch_size=512,
	validation_split=0.2,
)
#This way the model knows to hold out the 20% of the data to use it as validation data.

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "r--", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("[IMDB] Training and validation loss")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Loss")
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)

print(f"\nTest Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]*100:.2f}%")

predictions = model.predict(x_test)
print(predictions[:50])