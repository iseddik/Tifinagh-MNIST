# CNN - Tifinagh-MNIST

## Libraries
"""

import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.utils import *
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

"""## Data loading and adaptation """

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
x_train, y_train = upload_data('drive/MyDrive/DATA2/train_data/', n_class, n_train)
x_test, y_test = upload_data('drive/MyDrive/DATA2/test_data/', n_class, n_test)


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

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = to_categorical(y_train, n_class)
y_test = to_categorical(y_test, n_class)

"""## Architecture of the model"""

def define_model(input_size = (28, 28, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(128, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(128, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    

    conv3 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(32, 3, activation='relu', padding='same')(pool3)
    
    fltt = Flatten()(conv4)
    
    dan = Dense(33, activation='softmax')(fltt)
    
    model = Model(inputs=inputs, outputs=dan)
    
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
   
    
    return model

model = define_model((28, 28, 1))
model.summary()

his = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

"""## Model prediction on test data after training"""

def plot_predictions(model, num=3):
    fig, axes = plt.subplots(1, num, figsize=(12, 8))
    for i in range(num):
        index = np.random.randint(len(y_test))
        pred = np.argmax(model.predict(np.reshape(x_test[index], (1, 28, 28))))
        axes[i].imshow(np.reshape(x_test[index], (28, 28)))
        axes[i].set_title('Predicted label: '+ str(pred) + '\n/ true label :'+ str([e for e, x in enumerate(y_test[index]) if x == 1][0]))
        axes[i].axis('off')
    
    plt.show()


plot_predictions(model, num=5)
score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

import matplotlib.pyplot as plt
import numpy as np
with plt.xkcd():
    plt.plot(his.history['accuracy'], color='c')
    plt.plot(his.history['val_accuracy'], color='red')
    plt.title('Tifinagh-MNIST model accuracy')
    plt.legend(['acc', 'val_acc'])
    plt.savefig('acc_Tifinagh_MNIST_cnn.png')
    plt.show()

with plt.xkcd():
    plt.plot(his.history['loss'], color='c')
    plt.plot(his.history['val_loss'], color='red')
    plt.title('Tifinagh-MNIST model loss')
    plt.legend(['loss', 'val_loss'])
    plt.savefig('loss_Tifinagh_MNIST_cnn.png')
    plt.show()