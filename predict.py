#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:05 2020

@author: sudhanshukumar
"""

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class objClassification:
    def __init__(self,filename):
        self.filename =filename


    def predictionObj(self):
        # load model
        model = load_model('mymodel_VggNet.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        result = np.argmax(result)

        if result == 0:
            prediction = 'Alexa Device'
        elif result == 1:
            prediction = 'Anup'
        elif result == 2:
            prediction = 'Mug'
        elif result == 3:
            prediction = 'Papa'
        elif result == 4:
            prediction = 'Pradip'
        elif result == 5:
            prediction = 'Rahul'
        elif result == 6:
            prediction = 'Shubham'
        elif result == 7:
            prediction = 'Tejas'
        elif result == 8:
            prediction = 'Mummy'
        return [{"image": prediction}]

