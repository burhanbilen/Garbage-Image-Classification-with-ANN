from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

train_path='./Dataset/train/'
test_path='./Dataset/test/'

train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.25,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='wrap',
    validation_split=0.1)

val_datagen=ImageDataGenerator(rescale=1./255, validation_split=0.1)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    train_path,
    shuffle = True,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=64,
    subset='training')

val_generator=val_datagen.flow_from_directory(
    train_path,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=64,
    subset='validation')

test_generator=test_datagen.flow_from_directory(
    test_path,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=1)

print(test_generator.class_indices)
print(test_generator.classes)

model=Sequential()
model.add(Flatten(input_shape=(150,150,3)))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024,activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128,activation='relu'))
model.add(Dense(6,activation='softmax'))

adam = Adam(learning_rate=1e-5)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])

model.summary()

history=model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples//64,
    validation_steps=val_generator.samples//64,
    validation_data=val_generator,
    epochs=250)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()

y_true = test_generator.classes
pred = model.predict_classes(test_generator)

sns.heatmap(confusion_matrix(y_true, pred), annot= True)
plt.show()

loss, accuracy = model.evaluate(test_generator)
print("\n", loss, accuracy)
