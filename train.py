import os
import tensorflow as tf
import cv2
import glob
import numpy as np

x_train = []
y_train = []
x_test = []
y_test = []
x_train_len = len(next(os.walk("./dataset/train"))[1])
for i, j in enumerate(os.walk("./dataset/train/")):
    if i == 0:
        continue
    for k in glob.iglob(os.path.normpath(j[0]) + "/*.jpg"):
        length = [0] * 5
        length[i - 1] = 1
        img = cv2.imread(k)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            x_train.append(img)
            y_train.append(length)
        else:
            print(f"Error reading image: {k}")

# x_test_len = len(next(os.walk("./dataset/test"))[1])
# for i, j in enumerate(os.walk("./dataset/test/")):
#     if i == 0:
#         continue
#     for k in glob.iglob(os.path.normpath(j[0]) + "/*.jpg"):
#         length = [0] * x_test_len
#         length[i - 1] = 1
#         img = cv2.imread(k)
#         if img is not None:
#             img = cv2.resize(img, (128, 128))
#             x_test.append(img)
#             y_test.append(length)
#         else:
#             print(f"Error reading image: {k}")

if len(x_train) == 0 or len(y_train) == 0:
    raise ValueError("No training data available.")

print(x_train, y_train)


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(128, 128, 3)
        )
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(128, 128, 3)
        )
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)


y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
x_train = np.array(x_train) / 255.0

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)

print(x_train.shape, y_train.shape)
input_shape = x_train.shape[1:]

model = Model()
model.build((None, *input_shape))
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=["accuracy"],
)

try:
    model.fit(x_train, y_train, epochs=20)
except Exception as err:
    print("Error training: ")
    print(err)

# model.summary()

print("Running Successfully")
