import os
import shutil
import tensorflow as tf
from keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt

epochs_num=50

base_dir='E:\\linear_algebre\\distinguish_10_things'

train_dir=os.path.join(base_dir,'source','train')
test_dir=os.path.join(base_dir,'source','test')

train_dategen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.00,
                                                            horizontal_flip=True)

train_generator=train_dategen.flow_from_directory(
    train_dir,
    target_size=(32,32),
    batch_size=1024,
    class_mode='categorical'
)

test_dategen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.00)

test_generator=test_dategen.flow_from_directory(
    test_dir,
    target_size=(32,32),
    batch_size=1024,
    class_mode='categorical'
)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (32,32,3)))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Conv2D(256, (3,3), activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(1024, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(1024, activation = 'relu'))

model.add(tf.keras.layers.Dense(1024, activation = 'relu'))

model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])

history = model.fit(
                    train_generator,
                    epochs=epochs_num, 
                    validation_data=test_generator,
                    verbose=1
                    )

print(history.history)
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

import time
nowtime = time.strftime('%Y_%m_%d_%H_%M_%S')
mode_dir=os.path.join(base_dir,'mode',str(nowtime)+'(val_acc='+str(round(val_acc[epochs_num-1]*100)/100)+')')
os.mkdir(mode_dir)
model.save(os.path.join(mode_dir,'mode.h5'))

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs_num)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(os.path.join(mode_dir,'acc='+str(round(acc[epochs_num-1]*100)/100)+'  val_acc='+str(round(val_acc[epochs_num-1]*100)/100)+'.png'))
script_path=os.path.join(base_dir,'code','main.py')
target_path=os.path.join(mode_dir,'code.py')
shutil.copy(script_path, target_path)
