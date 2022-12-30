import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import *
from keras.layers import Input
from tensorflow.keras.models import Model
from keras.layers.pooling.max_pooling2d import MaxPool2D
from keras.layers.core.activation import Activation
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from numpy.core.shape_base import block
from keras.layers.convolutional.conv2d_transpose import Conv2D


def DiscriminativeSubNetwork(input_shape):
  # Initialize model
  base_channels = 64
  kernel_size = 3
  inputs = Input(shape=input_shape)
  #Encoder Block
  # block 1
  block_1 = Conv2D(filters=base_channels, kernel_size = kernel_size, input_shape=input_shape[1:],padding='same')(inputs)
  block_1 = BatchNormalization()(block_1)
  block_1 = Activation('relu')(block_1)
  block_1 = Conv2D(filters=base_channels, kernel_size = kernel_size, padding='same')(block_1)
  block_1 = BatchNormalization()(block_1)
  block_1 = Activation('relu')(block_1)
  block_1_mp = MaxPool2D(pool_size=(2, 2))(block_1)
  #block_2
  block_2 = Conv2D(filters=base_channels*2, kernel_size = kernel_size, padding='same')(block_1_mp)
  block_2 = BatchNormalization()(block_2)
  block_2 = Activation('relu')(block_2)
  block_2 = Conv2D(filters=base_channels*2, kernel_size = kernel_size, padding='same')(block_2)
  block_2 = BatchNormalization()(block_2)
  block_2 = Activation('relu')(block_2)
  block_2_mp = MaxPool2D(pool_size=(2, 2))(block_2)
  #block_3
  block_3 = Conv2D(filters=base_channels*4, kernel_size = kernel_size, padding='same')(block_2_mp)
  block_3 = BatchNormalization()(block_3)
  block_3 = Activation('relu')(block_3)
  block_3 = Conv2D(filters=base_channels*4, kernel_size = kernel_size, padding='same')(block_3)
  block_3 = BatchNormalization()(block_3)
  block_3 = Activation('relu')(block_3)
  block_3_mp = MaxPool2D(pool_size=(2, 2))(block_3)
  #block_4
  block_4 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_3_mp)
  block_4 = BatchNormalization()(block_4)
  block_4 = Activation('relu')(block_4)
  block_4 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_4)
  block_4 = BatchNormalization()(block_4)
  block_4 = Activation('relu')(block_4)
  block_4_mp = MaxPool2D(pool_size=(2, 2))(block_4)
  # block_5
  block_5 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_4_mp)
  block_5 = BatchNormalization()(block_5)
  block_5 = Activation('relu')(block_5)
  block_5 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_5)
  block_5 = BatchNormalization()(block_5)
  block_5 = Activation('relu')(block_5)
  block_5_mp = MaxPool2D(pool_size=(2, 2))(block_5)
  # block_6
  block_6 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_5_mp)
  block_6 = BatchNormalization()(block_6)
  block_6 = Activation('relu')(block_6)
  block_6 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_6)
  block_6 = BatchNormalization()(block_6)
  #decoder Block
  block_7 = UpSampling2D(size = 2,)(block_6)
  block_7 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_7)
  block_7 = BatchNormalization()(block_7)
  block_7 = Activation('relu')(block_7)
  block_7 = Concatenate(axis=3)([block_7, block_5])
  block_7 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_7)
  block_7 = BatchNormalization()(block_7)
  block_7 = Activation('relu')(block_7)
  block_7 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_7)
  block_7 = BatchNormalization()(block_7)
  block_7 = Activation('relu')(block_7)
  #block_8
  block_8 = UpSampling2D(size = 2,)(block_7)
  block_8 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_8)
  block_8 = BatchNormalization()(block_8)
  block_8 = Activation('relu')(block_8)
  block_8 = Concatenate(axis=3)([block_8, block_4])
  block_8 = Conv2D(filters=base_channels*4, kernel_size = kernel_size, padding='same')(block_8)
  block_8 = BatchNormalization()(block_8)
  block_8 = Activation('relu')(block_8)
  block_8 = Conv2D(filters=base_channels*4, kernel_size = kernel_size, padding='same')(block_8)
  block_8 = BatchNormalization()(block_8)
  block_8 = Activation('relu')(block_8)
  #block_9
  block_9 = UpSampling2D(size = 2,)(block_8)
  block_9 = Conv2D(filters=base_channels*4, kernel_size = kernel_size, padding='same')(block_9)
  block_9 = BatchNormalization()(block_9)
  block_9 = Activation('relu')(block_9)
  block_9 = Concatenate(axis=3)([block_9, block_3])
  block_9 = Conv2D(filters=base_channels*2, kernel_size = kernel_size, padding='same')(block_9)
  block_9 = BatchNormalization()(block_9)
  block_9 = Activation('relu')(block_9)
  block_9 = Conv2D(filters=base_channels*2, kernel_size = kernel_size, padding='same')(block_9)
  block_9 = BatchNormalization()(block_9)
  block_9 = Activation('relu')(block_9)
  #block_10
  block_10 = UpSampling2D(size = 2,)(block_9)
  block_10 = Conv2D(filters=base_channels*2, kernel_size = kernel_size, padding='same')(block_10)
  block_10 = BatchNormalization()(block_10)
  block_10 = Activation('relu')(block_10)
  block_10 = Concatenate(axis=3)([block_10, block_2])
  block_10 = Conv2D(filters=base_channels, kernel_size = kernel_size, padding='same')(block_10)
  block_10 = BatchNormalization()(block_10)
  block_10 = Activation('relu')(block_10)
  block_10 = Conv2D(filters=base_channels, kernel_size = kernel_size, padding='same')(block_10)
  block_10 = BatchNormalization()(block_10)
  block_10 = Activation('relu')(block_10)
  #block_11
  block_11 = UpSampling2D(size = 2,)(block_10)
  block_11 = Conv2D(filters=base_channels, kernel_size = kernel_size, padding='same')(block_11)
  block_11 = BatchNormalization()(block_11)
  block_11 = Activation('relu')(block_11)
  block_11 = Concatenate(axis=3)([block_11, block_1])
  block_11 = Conv2D(filters=base_channels, kernel_size = kernel_size, padding='same')(block_11)
  block_11 = BatchNormalization()(block_11)
  block_11 = Activation('relu')(block_11)
  block_11 = Conv2D(filters=base_channels, kernel_size = kernel_size, padding='same')(block_11)
  block_11 = BatchNormalization()(block_11)
  block_11 = Activation('relu')(block_11)

  #output
  outputs = Conv2D(filters=2, kernel_size = kernel_size, padding='same')(block_11)
  model = Model(inputs=[inputs], outputs=[outputs])
  return model


def ReconstructiveSubNetwork(input_shape):
  # Initialize model
  base_channels = 128
  kernel_size = 3
  inputs = Input(shape=input_shape)
  #Encoder Block
  # block 1
  block_1 = Conv2D(filters=base_channels, kernel_size = kernel_size, input_shape=input_shape,padding='same')(inputs)
  block_1 = BatchNormalization()(block_1)
  block_1 = Activation('relu')(block_1)
  block_1 = Conv2D(filters=base_channels, kernel_size = kernel_size, padding='same')(block_1)
  block_1 = BatchNormalization()(block_1)
  block_1 = Activation('relu')(block_1)
  block_1_mp = MaxPool2D(pool_size=(2, 2))(block_1)
  #block_2
  block_2 = Conv2D(filters=base_channels*2, kernel_size = kernel_size, padding='same')(block_1_mp)
  block_2 = BatchNormalization()(block_2)
  block_2 = Activation('relu')(block_2)
  block_2 = Conv2D(filters=base_channels*2, kernel_size = kernel_size, padding='same')(block_2)
  block_2 = BatchNormalization()(block_2)
  block_2 = Activation('relu')(block_2)
  block_2_mp = MaxPool2D(pool_size=(2, 2))(block_2)
  #block_3
  block_3 = Conv2D(filters=base_channels*4, kernel_size = kernel_size, padding='same')(block_2_mp)
  block_3 = BatchNormalization()(block_3)
  block_3 = Activation('relu')(block_3)
  block_3 = Conv2D(filters=base_channels*4, kernel_size = kernel_size, padding='same')(block_3)
  block_3 = BatchNormalization()(block_3)
  block_3 = Activation('relu')(block_3)
  block_3_mp = MaxPool2D(pool_size=(2, 2))(block_3)
  #block_4
  block_4 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_3_mp)
  block_4 = BatchNormalization()(block_4)
  block_4 = Activation('relu')(block_4)
  block_4 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_4)
  block_4 = BatchNormalization()(block_4)
  block_4 = Activation('relu')(block_4)
  block_4_mp = MaxPool2D(pool_size=(2, 2))(block_4)
  # block_5
  block_5 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_4_mp)
  block_5 = BatchNormalization()(block_5)
  block_5 = Activation('relu')(block_5)
  block_5 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_5)
  block_5 = BatchNormalization()(block_5)
  block_5 = Activation('relu')(block_5)
  #block_6
  block_6 = UpSampling2D(size = 2,)(block_5)
  block_6 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_6)
  block_6 = BatchNormalization()(block_6)
  block_6 = Activation('relu')(block_6)
  block_6 = Conv2D(filters=base_channels*8, kernel_size = kernel_size, padding='same')(block_6)
  block_6 = BatchNormalization()(block_6)
  block_6 = Activation('relu')(block_6)
  block_6 = Conv2D(filters=base_channels*4, kernel_size = kernel_size, padding='same')(block_6)
  block_6 = BatchNormalization()(block_6)
  block_6 = Activation('relu')(block_6)
  #block_7
  block_7 = UpSampling2D(size = 2,)(block_6)
  block_7 = Conv2D(filters=base_channels*4, kernel_size = kernel_size, padding='same')(block_7)
  block_7 = BatchNormalization()(block_7)
  block_7 = Activation('relu')(block_7)
  block_7 = Conv2D(filters=base_channels*4, kernel_size = kernel_size, padding='same')(block_7)
  block_7 = BatchNormalization()(block_7)
  block_7 = Activation('relu')(block_7)
  block_7 = Conv2D(filters=base_channels*2, kernel_size = kernel_size, padding='same')(block_7)
  block_7 = BatchNormalization()(block_7)
  block_7 = Activation('relu')(block_7)
  #block_8
  block_8 = UpSampling2D(size = 2,)(block_7)
  block_8 = Conv2D(filters=base_channels*2, kernel_size = kernel_size, padding='same')(block_8)
  block_8 = BatchNormalization()(block_8)
  block_8 = Activation('relu')(block_8)
  block_8 = Conv2D(filters=base_channels*2, kernel_size = kernel_size, padding='same')(block_8)
  block_8 = BatchNormalization()(block_8)
  block_8 = Activation('relu')(block_8)
  block_8 = Conv2D(filters=base_channels, kernel_size = kernel_size, padding='same')(block_8)
  block_8 = BatchNormalization()(block_8)
  block_8 = Activation('relu')(block_8)
  #block_9
  block_9 = UpSampling2D(size = 2,)(block_8)
  block_9 = Conv2D(filters=base_channels, kernel_size = kernel_size, padding='same')(block_9)
  block_9 = BatchNormalization()(block_9)
  block_9 = Activation('relu')(block_9)
  block_9 = Conv2D(filters=base_channels, kernel_size = kernel_size, padding='same')(block_9)
  block_9 = BatchNormalization()(block_9)
  block_9 = Activation('relu')(block_9)
  block_9 = Conv2D(filters=base_channels, kernel_size = kernel_size, padding='same')(block_9)
  block_9 = BatchNormalization()(block_9)
  block_9 = Activation('relu')(block_9)
  #output
  outputs = Conv2D(filters=3, kernel_size = kernel_size, padding='same')(block_9)
  model = Model(inputs=[inputs], outputs=[outputs])
  return model


# input_shape = (256,256, 3)
# model = ReconstructiveSubNetwork(input_shape)
#
# # model.build(input_shape)
# model.summary()
