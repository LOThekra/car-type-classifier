#!/usr/bin/env python
# coding: utf-8

# # Car type (SUV/Micro) classifier 
# This code classify the type of the car (suv/micro) using CNN and MobileNet

# In[1]:


import tensorflow as tf
tf.random.set_seed(7) 
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


# # Data preprocessing

# In[2]:


#Rescale images to 1/255
train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

#Resize the images to 150x150
#Generates batches of augmented data
#Use binary labels

train_dataset = train.flow_from_directory("C:/Users/Zekaro/Desktop/car_type/train/",
                                          target_size=(150,150),
                                          batch_size = 25,
                                          class_mode = 'binary')
                                         
test_dataset = test.flow_from_directory("C:/Users/Zekaro/Desktop/car_type/test/",
                                          target_size=(150,150),
                                          batch_size =10,
                                          class_mode = 'binary')
                                         


# In[3]:


#check the encoded class
test_dataset.class_indices


# # CNN model

# In[4]:


cnn_model = tf.keras.models.Sequential([
    #First convolution layer
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    #Second convolution layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #Third convoloution layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #Fourth convoloution layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #Flatten layer
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    #Sigmoid function for binary classification
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# The model will be trained using
# 1.   'adam' optimizer
# 2.   'binary_crossentropy' loss
# 3. 'accuracy' for metrics so the model will mnitor the accuracy during training

# In[5]:


cnn_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[6]:


#steps_per_epoch = train_imagesize/batch_size
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

cnn_history = cnn_model.fit(
      train_dataset,
      steps_per_epoch=3,  
      epochs=10,
      verbose=1,
      validation_data = test_dataset,
      )


# In[7]:


def predictImage(filename):
    #load the image and resize it to fit the shape
    img1 = image.load_img(filename,target_size=(150,150))
    plt.imshow(img1)
 
    #conver the image into numpy array 
    #expand the dimension of the array
    X = image.img_to_array(img1)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])

    #predict the value of the model
    val = cnn_model.predict(images)
    
    #print the prediction value
    if val[0] >= 0.5:
        print("The car type is SUV")   
    elif val[0] < 0.5:
        print("The car type is Micro")   


# In[8]:


predictImage('C:/Users/Zekaro/Downloads/index2.jpg')


# In[9]:


predictImage('C:/Users/Zekaro/Downloads/micro2.jpeg')


# # MobileNet model

# In[10]:


from tensorflow.keras.applications import MobileNetV2
mobile_model = Sequential()

#Add the MobileNet model
mobile_model.add(MobileNetV2(include_top = False, weights="imagenet",
                             input_shape=(150, 150, 3)))
#add a GlobalAveragePooling2D layer to reduce the size of the output
mobile_model.add(tf.keras.layers.GlobalAveragePooling2D())
#Sigmoid function for binary classification
mobile_model.add(Dense(1, activation = 'sigmoid'))


# The model will be trained using
# 1.   'adam' optimizer
# 2.   'binary_crossentropy' loss
# 3. 'accuracy' for metrics so the model will mnitor the accuracy during training

# In[11]:


mobile_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[12]:


#steps_per_epoch = train_imagesize/batch_size

mobile_history = mobile_model.fit(
      train_dataset,
      steps_per_epoch=3,  
      epochs=10,
      verbose=1,
      validation_data = test_dataset,
      )


# In[13]:


def predictImageMobile(filename):
    #load the image and resize it to fit the shape
    img1 = image.load_img(filename,target_size=(150,150))
    plt.imshow(img1)
 
   #conver the image into numpy array 
    #expand the dimension of the array
    X = image.img_to_array(img1)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])

    #predict the value of the model
    val = mobile_model.predict(images)

    #print the prediction value
    if val[0] >= 0.5:
        print("The car type is SUV")   
    elif val[0] < 0.5:
        print("The car type is Micro")     


# In[14]:


predictImageMobile('C:/Users/Zekaro/Downloads/index2.jpg')


# In[15]:


predictImageMobile('C:/Users/Zekaro/Downloads/micro2.jpeg')


# # Evaluation
# We will evaluate each model to get the accuracy for each

# In[16]:


cnn_model.evaluate(test_dataset)


# In[17]:


mobile_model.evaluate(test_dataset)


# AUC-ROC Curve is a performance measurement for classification problems that tells us how much a model is capable of distinguishing between classes. A higher AUC means that a model is more accurate.

# In[18]:


STEP_SIZE_TEST=test_dataset.n//test_dataset.batch_size
test_dataset.reset()

cnn_preds = cnn_model.predict(test_dataset,
                      verbose=1)

mobile_preds = mobile_model.predict(test_dataset,
                      verbose=1)


# In[19]:


from sklearn.metrics import roc_curve, auc

cnn_fpr, cnn_tpr, cnn_ = roc_curve(test_dataset.classes, cnn_preds)

mobile_fpr, mobile_tpr, mobile_ = roc_curve(test_dataset.classes, mobile_preds)


# In[20]:


cnn_roc_auc = auc(cnn_fpr, cnn_tpr)

mobile_roc_auc = auc(mobile_fpr, mobile_tpr)


# In[21]:


plt.figure()
lw = 2

plt.plot(cnn_fpr, cnn_tpr, color='darkorange',
         lw=lw, label='ROC curve-CNN (area = %0.2f)' % cnn_roc_auc)

plt.plot(mobile_fpr, mobile_tpr, color='green',
         lw=lw, label='ROC curve-MobileNet (area = %0.2f)' % mobile_roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




