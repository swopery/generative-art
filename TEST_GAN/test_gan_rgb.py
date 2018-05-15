"""
This code adds the ability to RGB images using the gan architecture

Important design decisions we have made for GAN:
1) Normalized inputs (-1,1)
2) Random sample from Guassian/Normal distribution (not uniform)
3) Separate batches of real and fake images
4) Avoid "sparse gradients" by using LeakyReLU/Avg Sampling (not ReLU/Max)
5) Adam optimizer

Reference: https://github.com/eriklindernoren/Keras-GAN
"""

# Keras Machine Learning tools
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD

# Tensorflow and Numpy
import tensorflow as tf
import numpy as np
from random import randint

# Special import of matplotlib/pyplot for Gattaca machine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Pillow for image manipulation
from PIL import Image
# glob for file paths/names
from glob import glob
import os

DIM_SIZE = 64
ALPHA = 0.1
PERCENT_FLIP = 4

class GAN():
    def __init__(self):
        # Define image specs
        self.img_rows = DIM_SIZE
        self.img_cols = DIM_SIZE
        self.channels = 3 #RGB
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Stochastic optimizer
        adam = Adam(0.0002, 0.5)

        # Build and compile first discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])

        #Build and compile the art discriminator
        self.art_disc = self.build_discriminator()
        self.art_disc.compile(loss='binary_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])

        # Build and compile generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy',
            optimizer=adam)

        # The generator takes noise as input of size 100x1, returns img
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.art_disc.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model (stacked generator and discriminator)
        # Input noise => generated images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy',
            optimizer=adam)

    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=ALPHA))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=ALPHA))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=ALPHA))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(2048))
        model.add(LeakyReLU(alpha=ALPHA))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()
        model.add(Flatten(input_shape=img_shape))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=ALPHA))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=ALPHA))

        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def get_image(self, image_path, width, height, mode):
        image = Image.open(image_path)
        if image.size != (width, height):
            new_w = image.size[0] - 50
            new_h = image.size[1] - 50
            j = (image.size[0] - new_w) // 2
            i = (image.size[1] - new_h) // 2
            image = image.crop([j,i,j+new_w, i + new_h])
            image = image.resize([width, height])
        return np.array(image.convert(mode))

    def get_batch(self, image_files, width, height, mode):
        # Creates massive Numpy array containing array for each image in dataset
        print("Resizing dataset...")
        data_batch = np.array(
            [self.get_image(sample_file, width, height, mode) for sample_file in image_files])

        return data_batch

    #trains the discriminators and generator
    def train(self, epochs, batch_size=128, save_interval=50):

        data_dir = './data'
        art_data_dir = './art_data'
        X_train = self.get_batch(glob(os.path.join(data_dir, '*.jpg')), DIM_SIZE, DIM_SIZE, 'RGB')
        A_train = self.get_batch(glob(os.path.join(art_data_dir, '*.jpg')), DIM_SIZE, DIM_SIZE, 'RGB')

        #Rescale -1 to 1 - rescales the 255 rgb values to range from -1 to 1 instead. normalized-good practice.
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        A_train = (A_train.astype(np.float32) - 127.5) / 127.5


        half_batch = int(batch_size / 2)

        #Create lists for logging the losses
        d_loss_logs_r = []
        d_loss_logs_f = []
        g_loss_logs = []
        da_loss_logs_r =[]
        da_loss_logs_f = []


        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            idx_a = np.random.randint(0, A_train.shape[0], half_batch)
            art_imgs = A_train[idx_a]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)


            # Train the discriminator with "noisy labeling"
            rand = randint(0,100)
            if rand < PERCENT_FLIP:
                d_loss_real = self.discriminator.train_on_batch(imgs, np.zeros((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.ones((half_batch, 1)))
            else:
                d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

            rand = randint(0,100)
            if rand < PERCENT_FLIP:
                da_loss_real = self.art_disc.train_on_batch(art_imgs, np.zeros((half_batch, 1)))
                da_loss_fake = self.art_disc.train_on_batch(gen_imgs, np.ones((half_batch, 1)))
            else:
                da_loss_real = self.art_disc.train_on_batch(art_imgs, np.ones((half_batch, 1)))
                da_loss_fake = self.art_disc.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            da_loss = 0.5 * np.add(da_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [DA_loss: %f, acc.: %.2f%%]" % (epoch, d_loss[0], 100*d_loss[1], g_loss, da_loss[0], 100*da_loss[1]))

            #Append the logs with the loss values in each training step
            d_loss_logs_r.append([epoch, d_loss[0]])
            d_loss_logs_f.append([epoch, d_loss[1]])
            g_loss_logs.append([epoch, g_loss])
            da_loss_logs_r.append([epoch, da_loss[0]])
            da_loss_logs_f.append([epoch, da_loss[1]])

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = (1/2.5) * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("output/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=75000, batch_size= 32, save_interval=100)

