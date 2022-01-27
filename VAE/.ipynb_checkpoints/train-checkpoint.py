import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import os
import argparse
from keras import backend as K
import tensorflow as tf
from keras.losses import mse
from tensorflow.compat.v1.keras.backend import set_session
K.clear_session()
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch, random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.compat.v1.Session(config=config))

def reparametrize(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_train = x_train/255
x_test = x_test.astype('float32')
x_test = x_test/255
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)



# Hyperparameters #
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--latent_dim", type=int, default=2, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--name", type=str, default='Model_1', help="Name of folder to save mdoels")
parser.add_argument("--name_img", type=str, default='traversal_vae', help="Name of file to save latent space traversal")
opt = parser.parse_args(args=[])
print(opt)


# Defining Encoder of VAE #
enc_inputs = keras.Input(shape=(28,28,1), name='Input')

x = layers.Conv2D(64, 3, strides=1, padding='same', name='Conv_1', activation='relu')(enc_inputs)
x = layers.BatchNormalization(name='BatchNorm_1')(x)
x = layers.Conv2D(64, 3, strides=2, padding='same', name='Conv_2', activation='relu')(x)
x = layers.BatchNormalization(name='BatchNorm_2')(x)
x = layers.Conv2D(128, 3, strides=2, padding='same', name='Conv_3', activation='relu')(x)
x = layers.BatchNormalization(name='BatchNorm_3')(x)
x = layers.Flatten()(x)
x = layers.Dense(100, activation='relu',name='Dense_1')(x)

mu = layers.Dense(opt.latent_dim, name='Mu')(x)
log_sigma = layers.Dense(opt.latent_dim, name='Log_sigma')(x)
z = layers.Lambda(reparametrize, output_shape=(opt.latent_dim,), name='z')([mu, log_sigma])

encoder = keras.Model(inputs=[enc_inputs], outputs=[mu, log_sigma, z], name='Encoder')
print(encoder.summary())


# Defining Decoder of VAE #

dec_inputs = keras.Input(shape=(opt.latent_dim,), name='Input')
x = layers.Dense(100, activation='relu',name='Dense_1')(dec_inputs)
x = layers.Dense(128*7*7, activation='relu',name='Dense_2')(x)
x = layers.Reshape((7,7,128))(x)
x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', name='Conv_1', activation='relu')(x)
x = layers.BatchNormalization(name='BatchNorm_1')(x)
x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', name='Conv_2', activation='relu')(x)
x = layers.BatchNormalization(name='BatchNorm_2')(x)
dec_outputs = layers.Conv2DTranspose(1, 3, strides=1, padding='same', name='Conv_3', activation='sigmoid')(x)

decoder = keras.Model(inputs=[dec_inputs], outputs=[dec_outputs], name='Decoder')
print(decoder.summary())

vae_outputs = decoder(encoder(enc_inputs)[2])
Vae = keras.Model(inputs=[enc_inputs], outputs=[vae_outputs], name='Vae')

mse_loss = mse(K.flatten(enc_inputs), K.flatten(vae_outputs))

kl_div = 1 + log_sigma - K.square(mu) - K.exp(log_sigma)  ## here we assume log_sigma is encoding (sigma)^2 so as to fit in the VAE formula
kl_div = (-0.5)*K.sum(kl_div, axis=-1) # Sum over all latent dimensions


# So KL loss term is the sum, however the mse loss gives the mean over all pixels, so to get the sum, mse loss is multiplied by total number of pixels.
mse_loss *= opt.img_size*opt.img_size
total_loss = K.mean(kl_div + mse_loss)

Vae.add_loss(total_loss)
Vae.compile(optimizer=keras.optimizers.Adam(lr=opt.lr))
print(Vae.summary())

model = opt.name+'.h5'
Vae.fit(x_train, epochs=opt.epochs, batch_size=opt.batch_size)
checkpoint_path = model
Vae.save_weights(checkpoint_path)

Vae.load_weights(checkpoint_path)


z_mean, z_log_var, z = encoder.predict(x_test,batch_size=opt.batch_size)
random_ = np.random.permutation(len(z))


# Interpolating to generate the latent space traversal
ratios = np.linspace(0, 1, num=100)
vectors = list()

for i in range (9):
    vectors.append(z[random_[i]])
    for ratio in ratios:
        vectors.append((1.0 - ratio) * z[random_[i]] + ratio * z[random_[i+1]])
    vectors.append(z[random_[i+1]])

vec = np.stack(vectors)
interpolate = decoder.predict(vec)
interpolate = np.transpose(torch.from_numpy(interpolate),(0,3,1,2))
save_image(interpolate, opt.name_img+'_%d.png' % 1, nrow=30, normalize=True)

