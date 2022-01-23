import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
from keras.preprocessing.image import smart_resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier, KerasRegressor
import cv2
import glob
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator

#model training
dim = 60
size = (dim,dim)
image_list = []
label_list_binary = []
label_list_color = []
for filename in glob.glob('C:/Users/Jess/Documents/ChessVision/Training/Occupied/Black/*.jpg'):
    image=cv2.imread(filename)
    gray = cv2.resize(image, size, cv2.INTER_LINEAR)
    image_list.append(gray)
    label_list_color.append(1)
    
for filename in glob.glob('C:/Users/Jess/Documents/ChessVision/Training/Occupied/White/*.jpg'):
    image=cv2.imread(filename)
    gray = cv2.resize(image, size, cv2.INTER_LINEAR)
    image_list.append(gray)
    label_list_color.append(2)
    
for filename in glob.glob('C:/Users/Jess/Documents/ChessVision/Training/Empty/*.jpg'):
    image=cv2.imread(filename)
    gray = cv2.resize(image, size, cv2.INTER_LINEAR)
    image_list.append(gray)
    label_list_color.append(0)
    
image_array = np.zeros((len(image_list),dim,dim,3))
label = np.zeros(len(image_list))
for i in range(0, len(image_list)): 
    image_array[i] = image_list[i]
    label[i] = label_list_color[i]
    
    
x_train, x_test, t_train, t_test = train_test_split(image_array, label, test_size = 0.2)
#scale input features to 0-1 range
x_train = x_train/255.0
x_test = x_test/255.0  

x_train.reshape(-1, dim, dim, 1)
x_test.reshape(-1,dim,dim,1)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(60,60,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
model.summary()
history = model.fit(x_train, t_train, batch_size = 32,epochs = 100, validation_data = (x_test, t_test))
model_json = model.to_json()
with open("C:/Users/Jess/Documents/ChessVision/Training/model.json","w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save_weights("C:/Users/Jess/Documents/ChessVision/Training/model.h5")
print("Saved model to disk")
