import numpy as np
import pandas as pd
import keras
from tqdm import tqdm
from tensorflow.keras.layers import Reshape, BatchNormalization, UpSampling2D, Activation, Conv2D, Conv2DTranspose, Dropout, Input, Flatten, LeakyReLU, Dense, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

class Generator():
    def __init__(self, v, latent_dim=100):
        self.latent_dim = latent_dim
        self.sn  = None
        self.img_shape = (128, 128, 1)
        self.model = None
        self.define_generator(verb=v)

    def define_generator(self, verb):
        model = Sequential()

        model.add(Dense(128 * 32 * 32, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((32, 32, 128)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(1, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        if verb:
            print("\n\n===== Generator Model Summary ========\n")
            model.summary()
        noise = Input(shape=(self.latent_dim,))

        img = model(noise)
        self.model = Model(noise, img)
        return Model(noise, img)
