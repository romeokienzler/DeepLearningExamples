#Example taken from https://www.tensorflow.org/tutorials/

import time
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(512, input_shape=(784,),activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



start = time.time()
model.fit(x_train, y_train, epochs=5)
stop = time.time()

print("Training time: " + stop-start)
model.evaluate(x_test, y_test)
