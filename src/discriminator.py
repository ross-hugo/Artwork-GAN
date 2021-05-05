from tensorflow.keras.datasets import mnist #for testing on some data
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Activation, ZeroPadding2D, GlobalAveragePooling1D, UpSampling2D, Conv2D

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy import ndimage

#discriminator.py
#losses: rotation loss & hinge loss (for the true versus fake prediction)
#penalties (such as the gradient penalty)
#normalization techniques: self-modulated batch normalization which doesnt require labels
#neural architecture: ResNet
#evaluation metrics: FID score
#ResNet contains 6 blocks
#I'm assuming that we will have a GAN class and inside that class we'll have the discriminator and generator functions

class Discriminator():
    def __init__(self):
        pass

  #discriminator outputs likelihood of image being real
    def define_discriminator(self, verb, sample_shape):
        #as per paper batch normalization is omitted in the discriminator
        #leakyRelu / conv / leakyrelu / conv
        #4 resblocks --> relu, global sum pooling, dense

        model = Sequential()

        #block1
        # model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=sample_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))


        #block2
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))

        #block3
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, strides=2,  padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))

        #block4
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))

        #add ReLU layer
        model.add(LeakyReLU(alpha=0.2))

        #add global sum pooling layer (is sum pooling the same as average pooling?) - also what parameters to be added here?
        # model.add(GlobalAveragePooling1D(self.input_shape)) # (batch_size, steps, features)

        model.add(Flatten())

        if verb:
            print("\n\n===== Discriminator Model Summary ======\n") 
            model.summary()

        img = Input(shape=(128, 128, 3))

        # features = model(img)
        # valid = Dense(1, activation="sigmoid")(features)
        # label = Dense(self.num_classes+1, activation="softmax")(features)

        # #valid = model(img)
        # # return Model(img, [valid, label]) #return img and validity
        # #return Model(img, valid)

        # # from https://github.com/vandit15/Self-Supervised-Gans-Pytorch/blob/01408fcce3e6cf4795d90c0f9d27e6906d5b59f3/main.py

        

        # lr = 1e-4
        # betas = (.9, .99)
        # opt = Adam(learning_rate=lr, beta_1= betas[0], beta_2=betas[1])
        # model.compile(loss="binary_crossentropy", optimizer=opt)

        img = Input(shape=(128, 128, 3))
        features = model(img)
        valid = Dense(1, activation="sigmoid")(features)
        label = Dense(5, activation="softmax")(features)

        return Model(img, [valid, label])

        #return model

    def compile(self):
        optimizer = Adam(0.0002, 0.5)
        self.model.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer,
            metrics=['accuracy']
        )

    def train(self, epochs, batch_size=128, save_interval=500):
        # #load real images
        (X_train,_), (Y_train,_) = mnist.load_data() #we're not gonna use Y_train but it bothers me to leave it out

        #normalize data (-1 to 1)
        #if you want to do 0 to 1 change 127.5 to 255
        # X_train = (X_train.astype(np.float32)-127.5) /127.5
        # X_train = X_train.astype(np.float32)/255
        #add dimension, if input to gen and discr has shape 28x28x1, then 3 dimensions
        X_train = np.expand_dims(X_train, axis=3)
        half_size = int(batch_size/2)

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], half_size)
            imgs = X_train[idx]

            #half batch number of vectors, each of size 100
            noise = np.random.normal(0,1,(half_size, 100)) #for generator
            #generate half batch of fake images
            gen_imgs = generator.predict(noise)

            ########################
            #Training Discriminator#
            ########################
            #train discriminator on real images
            d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            #train discriminator on fake images
            d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_size,1)))
            #averaged loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #add and divide by 2


            ########################
            ###Training Generator###
            ########################
            #generating fake images
            noise = np.random.normal(0,1,(batch_size, 100))
            #telling discriminator the image is real
            valid_y = np.array([1] * batch_size)

            g_loss = combined.train_on_batch(noise, valid_y)
