import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
print(sys.version)
from tensorflow.keras.layers import Reshape, BatchNormalization, UpSampling2D, Activation, Conv2D, Conv2DTranspose, Dropout, Input, Flatten, LeakyReLU, Dense, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

class Generator():
    def __init__(self):
        pass

    def define_generator(self, verb, sample_shape, latent_dim=100):
        model = Sequential()

        model.add(Dense(128 * 32 * 32, activation="relu", input_dim=latent_dim))
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

        model.add(Conv2D(sample_shape[-1], kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        if verb:
            print("\n\n===== Generator Model Summary ========\n")
            model.summary()
        noise = Input(shape=(latent_dim,))

        img = model(noise)
        return Model(noise, img)
