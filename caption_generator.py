"""
This file contains code for sampling caption using the model built (default: sample_model.h5)
It will first convert all the images from the sample_images folder into a set of VGG16 features, 
and then pass the features to the trained deep-learning model and tokenizer,
and return the captions.
"""

import os
import keras
from keras.preprocessing.text import tokenizer_from_json
import numpy as np
import matplotlib.pyplot as plt
from build_model.build import feature_extractions, sample_caption
import json
from pickle import load, dump
    
#Load tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_json = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_json)
    
model = keras.models.load_model("./sample_model.h5") #Load model
vocab_size = tokenizer.num_words #The number of vocabulary
max_length = 37 #Maximum length of caption sequence

#sampling
features = feature_extractions("./sample_images")

for i, filename in enumerate(features.keys()):
    plt.figure(i+1)
    caption = sample_caption(model, tokenizer, max_length, vocab_size, features[filename])
    
    img = keras.preprocessing.image.load_img("./sample_images/{fn}.jpg".format(fn=filename))
    plt.imshow(img)
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)



