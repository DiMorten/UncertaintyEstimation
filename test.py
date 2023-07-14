#!/usr/bin/env python3

import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from IPython.display import clear_output

from dataset import Dataset
from neural_network_model import NNModel
import cv2
from tensorflow.keras.models import Model, load_model, Sequential
from evidential_learning import DirichletLayer
from evidential_learning import alpha_to_probability_and_uncertainty
import numpy as np
import matplotlib.pyplot as plt

dataset = None
nnmodel = None

def main():

    path_base = 'D:/Jorge/datasets/underwater/TEST/images/'
    name = 'd_r_84_.jpg'
    name = 'f_r_1033_.jpg'
    
    filename = os.path.join(path_base, name)
    print("filename", filename)
    im = cv2.imread(filename)
    im = cv2.resize(im, (256, 256))
    print("im.shape", im.shape)

    model = load_model('vgg16_model_cc.h5', custom_objects={"DirichletLayer": DirichletLayer },
                       compile=False)
    # model = load_model('saved_model_cc_an0/my_model', custom_objects={"DirichletLayer": DirichletLayer })

    alpha = model.predict(np.expand_dims(im, axis=0))
    alpha = np.squeeze(alpha)
    print("alpha.shape", alpha.shape)
    belief, u = alpha_to_probability_and_uncertainty(alpha)
    
    max_belief = np.max(belief, axis=-1)
    u_thresholded = u.copy()
    u_thresholded[u >= max_belief] = 1
    u_thresholded[u < max_belief] = 0
    

    print("np.min(u), np.mean(u), np.max(u)", np.min(u), np.mean(u), np.max(u))
    # plot 3 subplots horizontally
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
    ax1.imshow(im)
    ax1.set_title('Input image')
    ax2.imshow(np.argmax(belief, axis=-1))
    ax2.set_title('Belief')
    ax3.imshow(u)
    ax3.set_title('Uncertainty')
    ax4.imshow(u_thresholded)
    ax4.set_title('Thresholded uncertainty')
    plt.show()
    print("belief.shape", belief.shape)
    print("u.shape", u.shape)

if __name__ == "__main__":
    main()
