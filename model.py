from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt

img_width, img_height = 400, 400

train_data_dir = '/content/drive/My Drive/DatasetCrop/Train'
validation_data_dir = '/content/drive/My Drive/DatasetCrop/Test'
nb_train_samples = 5109
nb_validation_samples = 1320
epochs = 16
batch_size = 64

#update the input shape and channels dimention
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(16, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(16, (2, 2), kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (2, 2), kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (2, 2), kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(128, (2, 2), kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, kernel_initializer='random_uniform'))
model.add(Dense(64, kernel_constraint=max_norm(2.))) #model.add(Dense(64, kernel_constraint=max_norm(2.)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy']
              )

train_datagen = ImageDataGenerator(
          samplewise_center=True,
          horizontal_flip=True,
          rotation_range=360,
          rescale=1/255.)

test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  samplewise_center=True,)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='binary')

history = model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs, validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size)

print(model.summary())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.save_weights('model_saved.h5')