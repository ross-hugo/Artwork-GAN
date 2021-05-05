from tensorflow.keras.datasets import mnist #for testing on some data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from src.discriminator import Discriminator
from src.generator import Generator

class SGAN():
    def __init__(self, verbosity=True, latent_dim=100):
        img_shape = (128, 128, 3)

        the_disc = Discriminator()
        the_gen = Generator()
        self.discriminator = the_disc.define_discriminator(verb=verbosity, sample_shape=img_shape)
        self.generator = the_gen.define_generator(verb=verbosity, sample_shape=img_shape, latent_dim=latent_dim)
        self.discriminator.trainable = False

        optimizer = Adam(0.0002, 0.5)
        self.discriminator.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer,
            metrics=['accuracy']
        )

        noise = Input(shape=(latent_dim,))
        img = self.generator(noise)

        valid = self.discriminator(img)

        self.combined = Model(noise, valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)


    def train(self, epochs, batch_size, X=None, y=None, display_iter=10):
        #load real images
        if X != None and y != None:
            X_train, Y_train = X, y
        elif X == None and y == None:
            (X_train,_), (Y_train,_) = mnist.load_data() #we're not gonna use Y_train but it bothers me to leave it out
        else:
            raise Exception("You did something weird. Give me all X and y or None for X and y")

        #normalize data (-1 to 1)
        #if you want to do 0 to 1 change 127.5 to 255
        # X_train = (X_train.astype(np.float32)-127.5) /127.5
        # X_train = X_train.astype(np.float32)/255
        #add dimension, if input to gen and discr has shape 28x28x1, then 3 dimensions
        #X_train = np.expand_dims(X_train, axis=3)
        
        half_size = int(batch_size/2)

        for epoch in tqdm(range(epochs)):
            imgs = [x for x, y in X_train.take(1)]
            
            #idx = np.random.randint(0, X_train.shape[0], half_size)
            #imgs = X_train[idx]

            #half batch number of vectors, each of size 100
            noise = np.random.normal(0,1,(half_size, 100)) #for generator
            #generate half batch of fake images
            gen_imgs = self.generator.predict(noise)

            ########################
            #Training Discriminator#
            ########################
            #train discriminator on real images
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_size, 1)))
            #train discriminator on fake images
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_size,1)))
            #averaged loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #add and divide by 2


            ########################
            ###Training Generator###
            ########################
            #generating fake images
            noise = np.random.normal(0,1,(batch_size, 100))
            #telling discriminator the image is real
            valid_y = np.array([1] * batch_size)

            g_loss = self.combined.train_on_batch(noise, valid_y)

            if epoch % display_iter == 0:
                self.display_generated_images()

    def display_generated_images(self):
        noise = np.random.normal(0,1,(2, 100)) 
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 127.5 * (gen_imgs + 1)

        for i in range(2):
            ax = plt.subplot(1, 2, i+1)
            plt.imshow(gen_imgs[i].astype('uint8'))

    def compile(self):
        self.generator.compile()
        self.discriminator.compile()

