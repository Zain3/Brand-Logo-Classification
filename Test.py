#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('bmh')

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def test_function(data, labels):

    data = data.reshape(-1, 300, 300, 3)

    labels_names = ['Nike',
                    'Adidas',
                    'Ford',
                    'Honda',
                    'General Mills',
                    'Unilever',
                    "McDonald's",
                    'KFC',
                    'Gators',
                    '3M']


    from tensorflow import keras

    # Load saved model
    new_model = keras.models.load_model('final_model.h5')

    # Evaluate on test data
    loss, acc = new_model.evaluate(data, labels)

    print('The overall loss is', loss)
    print('The overall accuracy is', acc)

    # label predictions
    y_trained = np.argmax(new_model.predict(data),axis=1)
   
    print(classification_report(labels, y_trained, target_names=labels_names))

    


# Loading Data
data_test = np.load('datasets/data_test.npy')
labels_test = np.load('datasets/labels_test.npy')

test_function(data_test, labels_test)




