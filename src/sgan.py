from tensorflow.keras.datasets import mnist #for testing on some data

import numpy as np
from tqdm.notebook import tqdm

from src.discriminator import Discriminator
from src.generator import Generator

class SGAN():
    def __init__(self, verbosity=True):
        self.discriminator = Discriminator(v=verbosity)
        self.generator = Generator(v=verbosity)

    def train(self, epochs, batch_size, X=None, y=None):
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
            gen_imgs = self.generator.model.predict(noise)

            ########################
            #Training Discriminator#
            ########################
            #train discriminator on real images
            d_loss_real = self.discriminator.model.train_on_batch(imgs, np.ones((half_size, 1)))
            #train discriminator on fake images
            d_loss_fake = self.discriminator.model.train_on_batch(gen_imgs, np.zeros((half_size,1)))
            #averaged loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #add and divide by 2


            ########################
            ###Training Generator###
            ########################
            #generating fake images
            noise = np.random.normal(0,1,(batch_size, 100))
            #telling discriminator the image is real
            valid_y = np.array([1] * batch_size)

            g_loss = self.generator.model.train_on_batch(noise, valid_y)

    def compile(self):
        self.generator.compile()
        self.discriminator.compile()

