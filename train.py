'''
3D GAN using Keras (based on DCGAN) - written by Robbie Barrat (robbiebarrat@gmail.com)
Dependencies: tensorflow 1.0 and keras 2.0
'''

import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv3D, UpSampling3D, Conv3DTranspose
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt


# GLOBAL VARIABLES

numpy_array_saved = "" # path to your .npy file generated with binvox_dataset.py






class ElapsedTimer(object):
	def __init__(self):
		self.start_time = time.time()
	def elapsed(self,sec):
		if sec < 60:
			return str(sec) + " sec"
		elif sec < (60 * 60):
			return str(sec / 60) + " min"
		else:
			return str(sec / (60 * 60)) + " hr"
	def elapsed_time(self):
		print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
	def __init__(self, img_rows=32, img_cols=32, img_depth=32, channel=1):

		self.img_rows = img_rows
		self.img_cols = img_cols
		self.img_depth = img_depth
		self.channel = channel
		self.D = None   # discriminator
		self.G = None   # generator
		self.AM = None  # adversarial model
		self.DM = None  # discriminator model 

	def discriminator(self):
		if self.D:
			return self.D
		self.D = Sequential()
		depth = 50
		dropout = 0.4
		# In: 28 x 28 x 1, depth = 1
		# Out: 14 x 14 x 1, depth=64
		
		input_shape = (self.img_rows, self.img_cols, self.img_depth, self.channel)
		kernel = (5,5,5)
		strides = (2,2,2)
		self.D.add(Conv3D(depth*1, kernel, strides=strides, input_shape=input_shape,\
			padding='same'))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(dropout))

		self.D.add(Conv3D(depth*2, kernel, strides=strides, padding='same'))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(dropout))

		self.D.add(Conv3D(depth*4, kernel, strides=strides, padding='same'))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(dropout))

		self.D.add(Conv3D(depth*8, kernel, strides=(1,1,1), padding='same'))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(dropout))

		# Out: 1-dim probability
		self.D.add(Flatten())
		self.D.add(Dense(1))
		self.D.add(Activation('sigmoid'))
		self.D.summary()
		return self.D

	def generator(self):
		if self.G:
			return self.G
		self.G = Sequential()
		dropout = 0.4
		depth = 50 * 4
		dim = 8
		kernel = (5,5,5)
		strides = (2,2,2)
		
		# In: 32
		# Out: dim x dim x depth
		self.G.add(Dense(dim*dim*dim*depth, input_dim=32))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))
		self.G.add(Reshape((dim, dim, dim, depth)))
		self.G.add(Dropout(dropout))

		# In: dim x dim x depth
		# Out: 2*dim x 2*dim x depth/2
		self.G.add(UpSampling3D())
		self.G.add(Conv3DTranspose(int(depth/2), kernel, padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))

		self.G.add(UpSampling3D())
		self.G.add(Conv3DTranspose(int(depth/4), kernel, padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))

		self.G.add(Conv3DTranspose(int(depth/8), kernel, padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))

		# Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
		self.G.add(Conv3DTranspose(1, kernel, padding='same'))
		self.G.add(Activation('sigmoid'))
		self.G.summary()
		return self.G

	def discriminator_model(self):
		if self.DM:
			return self.DM
		optimizer = RMSprop(lr=0.0002, decay=6e-8)
		self.DM = Sequential()
		self.DM.add(self.discriminator())
		self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
			metrics=['accuracy'])
		return self.DM

	def adversarial_model(self):
		if self.AM:
			return self.AM
		optimizer = RMSprop(lr=0.0001, decay=3e-8)
		self.AM = Sequential()
		self.AM.add(self.generator())
		self.AM.add(self.discriminator())
		self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
			metrics=['accuracy'])
		return self.AM

class MNIST_DCGAN(object):
	def __init__(self):
		self.img_rows = 32
		self.img_cols = 32
		self.img_depth = 32
		self.channel = 1
		print "initializing data..."
		self.x_train = np.load(numpy_array_saved)
		print "data initialized"

		print "initializing model"
		self.DCGAN = DCGAN()
		self.discriminator =  self.DCGAN.discriminator_model()
		self.adversarial = self.DCGAN.adversarial_model()
		self.generator = self.DCGAN.generator()
		print "model initialized"
	
	def train(self, train_steps=50000, batch_size=64, save_interval=0):
		noise_input = None
		if save_interval>0:
			noise_input = np.random.uniform(-1.0, 1.0, size=[16, 32])
		for i in range(train_steps):
			print "training... " + str(i)
			images_train = self.x_train[np.random.randint(0,
				self.x_train.shape[0], size=batch_size), :, :, :]
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 32])
			print "Generating stuff from network..."
			images_fake = self.generator.predict(noise)
			#images_fake = np.squeeze(images_fake)
			#for x in images_fake:
			#	print x.shape
			#for x in images_train:
			#	print x.shape
			print "Concatenating real + Fake images"
			if i % 50 == 0:
				outfile = open("generations/generation-" + str(i), "w") 
				np.save(outfile, images_fake)
			
			
			x = np.concatenate((images_train, images_fake))
			y = np.ones([2*batch_size, 1])
			y[batch_size:, :] = 0
			print "loss stuff..."
			d_loss = self.discriminator.train_on_batch(x, y)

			y = np.ones([batch_size, 1])

			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 32])

			a_loss = self.adversarial.train_on_batch(noise, y)

			log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])

			log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])

			print(log_mesg)
			if i % 32 == 0 or i == 0:
				self.generator.save(str(i) + '_2_gen.h5')
				self.adversarial.save(str(i) + '_2_adv.h5')



if __name__ == '__main__':
	mnist_dcgan = MNIST_DCGAN()
	timer = ElapsedTimer()
	mnist_dcgan.train(train_steps=100000, batch_size=64, save_interval=1)
	timer.elapsed_time()
