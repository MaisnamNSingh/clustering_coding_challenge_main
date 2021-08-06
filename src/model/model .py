# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 22:37:47 2021

@author: NIRANJAN SINGH
"""
from time import time
import os

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU

from sklearn.cluster import KMeans
import sklearn.metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.manifold import TSNE


np.random.seed(2021)



class EncoderDecoderBlock(layers.Layer):

    def __init__(self,
                 dims,
                 activation,
                 init,
                 ):
        super(EncoderDecoderBlock, self).__init__()
        self.num_layers = dims
        self.layer_count = len(dims) - 1
        self.activation = activation
        self.init = init
        self.input_data = Input(shape=(self.num_layers[0],), name='input')
        self.encoder_layers = [
            layers.Dense(self.num_layers[i + 1], activation='relu') for i in range(self.layer_count - 1)]
        self.latent = layers.Dense(self.num_layers[-1], kernel_initializer=init)
        self.decoder_layers = [
            layers.Dense(
                self.num_layers[i], activation='relu', kernel_initializer=init)
            for i in range(self.layer_count-1, 0, -1)]

    def call(self, loss, optimizer, metrics):

        inp = tf.keras.Input(shape=(self.num_layers[0],), name='input')
        x = inp
        for i in range(self.layer_count-1):
            x = self.encoder_layers[i](x)
        encoded = self.latent(x)

        x = encoded
        for i in range(self.layer_count-1):
            x = self.decoder_layers[i](x)
        x = Dense(self.num_layers[0], kernel_initializer=self.init, name='decoder_0')(x)
        decoded = x
        autoencoder_model = Model(
            inputs=inp, outputs=decoded, name='autoencoder')
        encoder_model = Model(inputs=inp, outputs=encoded, name='encoder')

        return autoencoder_model, encoder_model


class ClusteringLayer(Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(name='clusters', shape=(
            self.n_clusters, input_dim), initializer='glorot_uniform')

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):

        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs,
                                                       axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0

        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))

        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
