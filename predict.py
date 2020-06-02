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
        model = load_model('model_family.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'Papa'
        elif result[0][1] == 1:
            prediction = 'Rahul'
        elif result[0][2] == 1:
            prediction = 'Shubham'
        elif result[0][3] == 1:
            prediction = 'Mummy'
        # elif result[0][4] == 1:
        #     prediction = 'Mouse'
        # elif result[0][5] == 1:
        #     prediction = 'Mug'
        # elif result[0][6] == 1:
        #     prediction = 'Pradip Shirke'
        # elif result[0][7] == 1:
        #     prediction = 'Rahul Nandanwar'
        # elif result[0][8] == 1:
        #     prediction = 'Shubham Nandanwar'
        # elif result[0][9] == 1:
        #     prediction = 'Spects'
        # elif result[0][10] == 1:
        #     prediction = 'Tejas Patil'
        # elif result[0][11] == 1:
        #     prediction = 'Mummy'
        return [{"image": prediction}]


