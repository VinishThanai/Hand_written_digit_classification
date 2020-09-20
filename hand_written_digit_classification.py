#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow')


# In[ ]:


get_ipython().system('pip install numpy')


# In[ ]:


import tensorflow as tf
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


mnist = tf.keras.datasets.mnist


# In[ ]:


# ab load kiya h mnist data
# data in the form of images of 28*28 


# In[ ]:


# ab 70% data ko train mei daal and 30% test ke liye


# In[ ]:


(x_train,y_train),(x_test,y_test) = mnist.load_data()


# In[ ]:


plt.imshow(x_train[0])
plt.show()


# In[ ]:



plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()


# In[ ]:


# converted this to grayscale for easy image processing


# In[ ]:


print(x_train[0])


# In[ ]:


#Normalization, so that all values are within the range of 0 and 1.

x_train= tf.keras.utils.normalize(x_train,axis=1)
x_test= tf.keras.utils.normalize(x_test,axis=1)


# In[ ]:


print(x_train[0])


# In[ ]:


model=tf.keras.models.Sequential() #a feed forward model
model.add(tf.keras.layers.Flatten()) 
#takes our 28x28 and makes it 1x784
# 28*28  wale ko flat kiya 


# In[ ]:


model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) #a simple fully connected layer
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) # our output layer. 10 units for 10 classes. Softmax for probability distribution


# In[ ]:


model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy', #how will we calculate the error to minimize the loss
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10)


# In[ ]:


val_loss,val_acc=model.evaluate(x_test,y_test)


# In[ ]:


print(val_loss)


# In[ ]:


print(val_acc)


# In[ ]:


model.save(r'C:\Users\Vinish thanai\Desktop\CS\hand_written_digit_classification')


# In[ ]:


# ab training khatam hui h


# In[ ]:


new_model=tf.keras.models.load_model(r'C:\Users\Vinish thanai\Desktop\CS\hand_written_digit_classification')
predictions=new_model.predict(x_test)


# In[ ]:


print(predictions[0])
import numpy as np
plt.imshow(x_test[30],cmap=plt.cm.binary)
plt.show()
print(np.argmax(predictions[30]))

