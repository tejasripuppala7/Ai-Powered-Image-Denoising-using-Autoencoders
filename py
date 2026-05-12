import tensorflow as tf
import numpy as np

(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

noise_factor = 0.1

X_train_noisy = X_train + noise_factor * np.random.normal(0, 1, X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(0, 1, X_test.shape)

X_train_noisy = np.clip(X_train_noisy, 0, 1)
X_test_noisy = np.clip(X_test_noisy, 0, 1)

from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

from tensorflow.keras import layers, models

input_img = layers.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2,2), padding='same')(x)

x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2,2), padding='same')(x)

x = layers.Conv2DTranspose(16, (3,3), strides=2, activation='relu', padding='same')(encoded)
x = layers.Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same')(x)

decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

model = models.Model(input_img, decoded)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='mse'
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    x_train, x_train,
    validation_data=(x_test, x_test),
    epochs=20,
    batch_size=128,
    callbacks=[lr_scheduler]
)

history = model.fit(
    X_train_noisy, X_train,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(X_test_noisy, X_test)
)

decoded_imgs = model.predict(X_test_noisy)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(18,5))

for i in range(n):

    ax = plt.subplot(3, n, i+1)
    plt.imshow(X_test[i].reshape(28,28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(X_test_noisy[i].reshape(28,28), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    ax = plt.subplot(3, n, i+1+2*n)
    img = decoded_imgs[i].reshape(28,28)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    plt.imshow(img, cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.show()

from google.colab import drive

drive.mount('/content/drive')

model.save('/content/drive/MyDrive/autoencoder_finallll.h5')

from google.colab import files
uploaded = files.upload()

import cv2
import numpy as np

filename = list(uploaded.keys())[0]

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img.astype("float32") / 255.0
img = img.reshape(1, 28, 28, 1)

output = model.predict(img)

import matplotlib.pyplot as plt

plt.imshow(img.reshape(28,28), cmap='gray')
plt.title("output Image")
plt.show()

print(history.history['loss'][-1])
print(history.history['val_loss'][-1])

model.save("model_1.keras")

print(type(model))

from tensorflow import keras

model = keras.models.load_model(
    "model.h5",
    compile=False
)

import numpy as np

img = np.array(img, dtype=np.float32)
img = img / 255.0
img = img.reshape(1, 28, 28, 1)

output = model.predict(img)

from google.colab import drive

drive.mount('/content/drive')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

X_train = np.random.rand(10000, 100)

model = Sequential([
    Dense(64, activation='relu', input_shape=(100,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(100, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
