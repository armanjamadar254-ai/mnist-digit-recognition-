# mnist-digit-recognition-
# MNIST Handwritten Digit Recognition

A neural network that reads handwritten digits (0 to 9).

## What I did
- Loaded 70,000 images of handwritten digits (60,000 for training, 10,000 for testing)
- Scaled pixel values from 0-255 down to 0-1
- Built a neural network with Keras (128 hidden neurons)
- Trained it for 5 epochs
- Looked at correct predictions and mistakes

## Results
- Training accuracy: 98.6%
- Test accuracy: 97.7% (on 10,000 images the model never saw)

## What I learned
- Scaling data helps neural networks learn
- A small gap between training and test accuracy is normal
- Most mistakes are on messy handwriting that is hard even for humans

## Tools
Python, TensorFlow/Keras, NumPy, Matplotlib, Google Colab

## Code
```python
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_s = X_train / 255.0
X_test_s = X_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train_s, y_train, epochs=5)

loss, acc = model.evaluate(X_test_s, y_test)
print("Test accuracy:", acc)

preds = model.predict(X_test_s)
pred_labels = np.argmax(preds, axis=1)

wrong = np.where(pred_labels != y_test)[0][:5]
plt.figure(figsize=(10, 3))
for j, i in enumerate(wrong):
    plt.subplot(1, 5, j + 1)
    plt.imshow(X_test[i], cmap="gray")
    plt.title(f"Pred: {pred_labels[i]}\nTrue: {y_test[i]}")
    plt.axis("off")
plt.show()
```