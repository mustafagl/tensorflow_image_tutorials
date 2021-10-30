# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:17:57 2021

@author: musta
"""

import numpy as np
import os
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import cv2
import pathlib
#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(r'C:\Users\musta\.keras\datasets\sgame_photos')

batch_size = 1
img_height = 64
img_width = 64

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.02,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.02,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(10).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

print(np.min(first_image), np.max(first_image))

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

num_classes = 3

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(6, 2, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(12, 2, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(24, 2, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(64, activation='relu'),


  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])




epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

img=cv2.imread(r'C:\Users\musta\.keras\datasets\sgame_photos\ucgen\ucgen0.png')
dim=(64,64)

img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])


print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

#cv2.imshow("asdas",img)
#cv2.waitKey()
#fnlimg=cv2.imread(r'C:\Users\musta\Desktop\sgame.jpg');

'''



window_name = 'Image'
thickness = 2
color = (255, 0, 0)
print(len(fnlimg[0]),len(fnlimg))

for n in range(1000):
    i=random.randint(0,len(fnlimg[0])-128)
    j=random.randint(0,len(fnlimg)-128)
    start_point = (i, j)
    end_point = (i+128, j+128)
    croppedimg=fnlimg[j:j+128,i:i+128]   
    
    resized = cv2.resize(croppedimg, dim)
    
    resized_arr = tf.keras.utils.img_to_array(resized)
    resized_arr = tf.expand_dims(resized_arr, 0)     
    predictions = model.predict(resized_arr)
    score = tf.nn.softmax(predictions[0])
    if class_names[np.argmax(score)]=="ucgen":
        rect = cv2.rectangle(fnlimg, start_point, end_point, color, thickness)
     

cv2.imshow(window_name, fnlimg)
cv2.waitKey()
'''



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
for images, labels in train_ds.take(1):
    print(class_names)


