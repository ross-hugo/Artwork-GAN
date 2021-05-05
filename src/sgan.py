import tensorflow as tf
from tensorflow.keras.datasets import mnist #for testing on some data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from src.discriminator import Discriminator
from src.generator import Generator

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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

        valid, _ = self.discriminator(img)

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
       
        half_size = int(batch_size/2)
        
        valid_weight = {0:0.5, 1:0.5}
        classes_weights = {i: 4 / half_size for i in range(4)}
        classes_weights[4] = 1/half_size

        #normalize data (-1 to 1)
        #if you want to do 0 to 1 change 127.5 to 255
        # X_train = (X_train.astype(np.float32)-127.5) /127.5
        # X_train = X_train.astype(np.float32)/255
        #add dimension, if input to gen and discr has shape 28x28x1, then 3 dimensions
        #X_train = np.expand_dims(X_train, axis=3)
        

        for epoch in tqdm(range(epochs)):
            imgs = [(y, 0, 1) for x, _ in X_train.take(1) for y in x.numpy()]


            nine_imgs = np.array([(np.rot90(x[0], 1), 1, 0) for x in imgs])


            one_eighty_imgs = [(np.rot90(x[0], 2), 2, 0) for x in imgs]
            two_seventy_imgs = [(np.rot90(x[0], 3), 3, 0) for x in imgs]
            
            #idx = np.random.randint(0, X_train.shape[0], half_size)
            #imgs = X_train[idx]

            full_list = np.concatenate((imgs, nine_imgs, one_eighty_imgs, two_seventy_imgs))
            random.shuffle(full_list)

            imgs, valid, classes, = [], [], []
            for x in full_list:
                imgs.append(x[0])
                classes.append(x[1])
                valid.append(x[2])
            imgs = (np.array(imgs) - 127.5)/127.5
            imgs, valid, classes = np.asarray(imgs).astype('float32'), np.asarray(valid).astype('uint8'), np.asarray(classes).astype('uint8')

            #half batch number of vectors, each of size 100
            noise = np.random.normal(0,1,(batch_size, 100)) #for generator
            #generate half batch of fake images
            gen_imgs = self.generator.predict(noise)

            ########################
            #Training Discriminator#
            ########################
            #train discriminator on real images
            #d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_size, 1)))
            classes = to_categorical(classes, num_classes=5)
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, classes], class_weight=[valid_weight, classes_weights])
            #train discriminator on fake images
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [np.zeros((batch_size,1)), 
                                                                        to_categorical(np.full((batch_size, 1), 4))], 
                                                                        class_weight=[valid_weight, classes_weights])
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

            # if epoch % display_iter == 0:
            #     self.display_generated_images()

    def display_generated_images(self):
        noise = np.random.normal(0,1,(1, 100)) 
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 127.5 * (gen_imgs + 1)

        for i in range(1):
            ax = plt.subplot(1, 2, i+1)
            plt.imshow(gen_imgs[i].astype('uint8'))
            plt.show()

    def compile(self):
        # self.generator.compile()
        # self.discriminator.compile()
        pass
