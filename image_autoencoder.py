#Takes image feature list and reduces its dimension

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
#from google.colab import drive
#drive.mount('/content/drive')


path = "saved/feat_list.npy"
features_path = "saved/deep_autoe_features.pickle"

#path = "/content/drive/My Drive/reco/feat_list.npy"
#features_path = '/content/drive/My Drive/reco/deep_autoe_features.pickle'

data = np.load(path)
data_size = len(data)
train_size = 0.8
encoding_dim = 300
input_img = Input(shape=(4096, ))
my_epochs = 50


# "encoded" is the encoded representation of the inputs
encoded = Dense(encoding_dim * 8, activation='relu')(input_img)
encoded = Dense(encoding_dim * 4, activation='relu')(encoded)
encoded = Dense(encoding_dim * 2, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
decoded = Dense(encoding_dim * 4, activation='relu')(decoded)
decoded = Dense(encoding_dim * 8, activation='relu')(decoded)
decoded = Dense(4096, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# Separate Encoder model

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# Separate Decoder model

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim, ))
# retrieve the layers of the autoencoder model
decoder_layer1 = autoencoder.layers[-4]
decoder_layer2 = autoencoder.layers[-3]
decoder_layer3 = autoencoder.layers[-2]
decoder_layer4 = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer4(decoder_layer3(decoder_layer2(decoder_layer1(encoded_input)))))

# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# prepare input data

means = []
ranges = []
def norm(x):
  means.append(x.mean())
  ranges.append((x.max() - x.min()))
  return ((x - x.mean())/(x.max() - x.min()))

def rev_norm(y, ind):
  return (y*ranges[ind] + means[ind])

norm_data = []
for i in range(len(data)):
  norm_data.append(norm(data[i]))

norm_data = np.array(norm_data)

x_train = norm_data[:int(data_size*train_size)]
x_test = norm_data[int(data_size*train_size):]

autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=256, shuffle=True, validation_data=(x_test, x_test),
                verbose=2)

encoded_imgs = encoder.predict(norm_data)
decoded_imgs = decoder.predict(encoded_imgs)

pickle.dump(encoded_imgs, open(features_path, 'wb'))
