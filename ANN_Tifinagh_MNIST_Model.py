# Classification by a classical ANN (MLP) - Tifinagh-MNIST

## The libraries we will use
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

"""## Data loading and adaptation"""

def upload_data(path_name, number_of_class, number_of_images): 
    X_Data = []
    Y_Data = []
    for i in range(number_of_class):
        images = os.listdir(path_name + str(i))
        for j in range(number_of_images):
            img = cv2.imread(path_name + str(i)+ '/' + images[j], 0)
            X_Data.append(img)
            Y_Data.append(i)
        print("> the " + str(i) + "-th file is successfully uploaded.", end='\r') 
    return np.array(X_Data), np.array(Y_Data)

n_class = 33
n_train = 2000
n_test = 500
#here we upload our data (Tifinagh data)
x_train, y_train = upload_data('/media/etabook/etadisk1/EducFils/PFE/DATA2/train_data/', n_class, n_train)
x_test, y_test = upload_data('/media/etabook/etadisk1/EducFils/PFE/DATA2/test_data/', n_class, n_test)


print("The x_train's shape is :", x_train.shape)
print("The x_test's shape is :", x_test.shape)
print("The y_train's shape is :", y_train.shape)
print("The y_test's shape is :", y_test.shape)

def plot_data(num=3):
    fig, axes = plt.subplots(1, num, figsize=(12, 8))
    for i in range(num):
        index = np.random.randint(len(x_test))
        axes[i].imshow(np.reshape(x_test[index], (28, 28)))
        axes[i].set_title('image label: %d' % y_test[index])
        axes[i].axis('off')
    
    plt.show()
        
plot_data(num=5)

plot_data(num=5)

num_classes = 33
size = 28

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (x_train.shape[0], size*size))
x_test = np.reshape(x_test, (x_test.shape[0], size*size))
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

"""## Define our neural network model (Architecture)"""

model = Sequential()
model.add(Dense(512, input_shape=(size*size,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.summary()

"""## Model prediction on test data before training  """

def plot_predictions(model, num=3):
    fig, axes = plt.subplots(1, num, figsize=(12, 8))
    for i in range(num):
        index = np.random.randint(len(x_test))
        pred = np.argmax(model.predict(np.reshape(x_test[index], (1, size*size))))
        axes[i].imshow(np.reshape(x_test[index], (size, size)))
        axes[i].set_title('Predicted label: '+ str(pred) + '\n/ true label :'+ str([e for e, x in enumerate(y_test[index]) if x == 1][0]))
        axes[i].axis('off')
    
    plt.show()
        
plot_predictions(model, num=5)

"""## Training"""

history = model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test, y_test))

"""## Model prediction on test data after training"""

plot_predictions(model, num=5)
score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""## Model history during training"""

import matplotlib.pyplot as plt
import numpy as np
with plt.xkcd():
    plt.plot(history.history['accuracy'], color='c')
    plt.plot(history.history['val_accuracy'], color='red')
    plt.title('Tifinagh-MNIST model accuracy')
    plt.legend(['acc', 'val_acc'])
    plt.savefig('acc_Tifinagh_MNIST.png')
    plt.show()

with plt.xkcd():
    plt.plot(history.history['loss'], color='c')
    plt.plot(history.history['val_loss'], color='red')
    plt.title('Tifinagh-MNIST model loss')
    plt.legend(['loss', 'val_loss'])
    plt.savefig('loss_Tifinagh_MNIST.png')
    plt.show()