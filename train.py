%load_ext tensorboard

import os
import random

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                     Dropout, Flatten, Input, Reshape,
                                     TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

tf.compat.v1.disable_eager_execution()

import midi

NUM_EPOCHS = 100
LR = 0.001
WRITE_HISTORY = True
NUM_RAND_SONGS = 10
DO_RATE = 0.1
BN_M = 0.9

BATCH_SIZE = 100
MAX_LENGTH = 16
PARAM_SIZE = 120

np.random.seed(0)
random.seed(0)

y_samples = np.load('samples.npy')
y_lengths = np.load('lengths.npy')
num_songs = y_lengths.shape[0]

y_shape = (num_songs, MAX_LENGTH) + y_samples.shape[1:]
y_orig = np.zeros(y_shape, dtype=y_samples.dtype)

cur_ix = 0
for i in range(num_songs):
	end_ix = cur_ix + y_lengths[i]
	for j in range(MAX_LENGTH):
		k = j % (end_ix - cur_ix)
		y_orig[i,j] = y_samples[cur_ix + k]
	cur_ix = end_ix

y_train = np.copy(y_orig)

y_test_song = np.copy(y_train[0:1])
midi.samples_to_midi(y_test_song[0], 'gt.mid', 16)

print("Building Model...")

x_in = Input(shape=y_shape[1:])
x = Reshape((y_shape[1], -1))(x_in)
x = TimeDistributed(Dense(2000, activation='relu'))(x)
x = TimeDistributed(Dense(200, activation='relu'))(x)
x = Flatten()(x)
x = Dense(1600, activation='relu')(x)
x = Dense(PARAM_SIZE)(x)
x = BatchNormalization(momentum=BN_M, name='pre_encoder')(x)
x = Dense(1600, name='encoder')(x)
x = BatchNormalization(momentum=BN_M)(x)
x = Activation('relu')(x)
if DO_RATE > 0:
	x = Dropout(DO_RATE)(x)
x = Dense(MAX_LENGTH * 200)(x)
x = Reshape((MAX_LENGTH, 200))(x)
x = TimeDistributed(BatchNormalization(momentum=BN_M))(x)
x = Activation('relu')(x)
if DO_RATE > 0:
	x = Dropout(DO_RATE)(x)
x = TimeDistributed(Dense(2000))(x)
x = TimeDistributed(BatchNormalization(momentum=BN_M))(x)
x = Activation('relu')(x)
if DO_RATE > 0:
	x = Dropout(DO_RATE)(x)
x = TimeDistributed(Dense(y_shape[2] * y_shape[3], activation='sigmoid'))(x)
x = Reshape((y_shape[1], y_shape[2], y_shape[3]))(x)

model = Model(x_in, x)
model.compile(optimizer=RMSprop(lr=LR), loss='binary_crossentropy')

###################################
#  Train
###################################
print("Compiling SubModels...")
func = K.function([model.get_layer('encoder').input, K.learning_phase()],
				  [model.layers[-1].output])
enc = Model(inputs=model.input, outputs=model.get_layer('pre_encoder').output)

rand_vecs = np.random.normal(0.0, 1.0, (NUM_RAND_SONGS, PARAM_SIZE))
np.save('rand.npy', rand_vecs)

def make_rand_songs(write_dir, rand_vecs):
	for i in range(rand_vecs.shape[0]):
		x_rand = rand_vecs[i:i+1]
		y_song = func([x_rand, 0])[0]
		midi.samples_to_midi(y_song[0], write_dir + 'rand' + str(i) + '.mid', 16, 0.25)

def make_rand_songs_normalized(write_dir, rand_vecs):
	x_enc = np.squeeze(enc.predict(y_orig))
	
	x_mean = np.mean(x_enc, axis=0)
	x_cov = np.cov((x_enc - x_mean).T)
	_, s, v = np.linalg.svd(x_cov)
	e = np.sqrt(s)

	print(f"Means: {x_mean[:6]}")
	print(f"Evals: {e[:6]} ")

	x_vecs = x_mean + np.dot(rand_vecs * e, v)
	make_rand_songs(write_dir, x_vecs)
	
		  
print("Training...")
train_loss = []
ofs = 0

for iter in range(NUM_EPOCHS):
	cur_ix = 0
	for i in range(num_songs):
		end_ix = cur_ix + y_lengths[i]
		for j in range(MAX_LENGTH):
			k = (j + ofs) % (end_ix - cur_ix)
			y_train[i,j] = y_samples[cur_ix + k]
		cur_ix = end_ix
	ofs += 1

	history = model.fit(y_train, y_train, batch_size=BATCH_SIZE, epochs=1)

	loss = history.history["loss"][-1]
	train_loss.append(loss)
	print(f"Train Loss: {str(train_loss[-1])}")
	
	i = iter + 1
	if i in [1, 10] or (i % 100 == 0):
		write_dir = 'History/e' + str(i)
		os.makedirs(write_dir)
		write_dir += '/'
		model.save('History/model.h5')
		print("Saved")

		y_song = model.predict(y_test_song, batch_size=BATCH_SIZE)[0]
		# util.samples_to_pics(write_dir + 'test', y_song)
		midi.samples_to_midi(y_song, write_dir + 'test.mid', 16)

		make_rand_songs_normalized(write_dir, rand_vecs)

print("Done")
