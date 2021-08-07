# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 16:55:31 2021

@author: NIRANJAN SINGH
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 21:59:53 2021

"""
import argparse
import os
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling

from sklearn.cluster import KMeans
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from model import EncoderDecoderBlock,ClusteringLayer


sys.path.append('../')
sys.path.append('./')

np.random.seed(1)
tf.random.set_seed(1)
batch_size = 128
epochs = 10
learning_rate = 1e-2
intermediate_dim = 64
original_dim = 784
n_clusters = 3
n_epochs   = 50
batch_size = 128
activation = 'relu'
loss='mse'
metrics = 'accuracy'
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                           distribution='uniform')
optimizer = SGD(lr=1, momentum=0.9)
pretrain_epochs = n_epochs
batch_size = batch_size
save_dir = './results'

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-f','--file', help='Description for foo argument', required=True)
    parser.add_argument('-p','--plot', help='Description for foo argument', required=False)
    args = vars(parser.parse_args())

    
    data = pd.read_csv(args['file'])
    data = data.drop('label',axis=1)
    print(data.head())
    
    numeric_columns = data.columns.values.tolist()
    scaler = MinMaxScaler() 
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    

    x = data.values
    dims = [x.shape[-1], 500, 500, 2000, 10]
    
    kmeans = KMeans(n_clusters=n_clusters, n_jobs=4)
     
    y_pred_kmeans = kmeans.fit_predict(x)
    
    autoencoder = EncoderDecoderBlock(dims,activation,init)
    autoencoder, encoder = autoencoder(loss, optimizer, metrics)
    
    
    from tensorflow.keras.utils import plot_model
    plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)
    from IPython.display import Image
    Image(filename='autoencoder.png') 
        
    autoencoder.compile(optimizer=optimizer, loss='mse')
    autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)
    autoencoder.save_weights('dan_1_weights.h5')
    
    
    from tensorflow.keras.utils import plot_model
    plot_model(encoder, to_file='encoder.png', show_shapes=True)
    from IPython.display import Image
    Image(filename='encoder.png')
        
    autoencoder.save_weights('dan_1_weights.h5')
    autoencoder.load_weights('dan_1_weights.h5')
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    
    
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    from IPython.display import Image
    Image(filename='model.png')
       
       
    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(x))
    y_pred_last = np.copy(y_pred)
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    # computing an auxiliary target distribution


    loss = 0
    index = 0
    maxiter = 1000 
    update_interval = 100 
    index_array = np.arange(x.shape[0])
    
    tol = 0.001 
    
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = model.predict(x, verbose=0)
            p = target_distribution(q)  
    
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model.train_on_batch(x=x[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
    
    model.save_weights('danaher_model_final.h5')
    model.load_weights('danaher_model_final.h5')
    # Eval.
    q = model.predict(x, verbose=0)
    p = target_distribution(q)  # update the auxiliary target distribution p
    
    # evaluate the clustering performance
    y_pred = q.argmax(1)
    df = data.copy()
    df['cluster'] = y_pred
    df['cluster'].value_counts()
    
    tsne = TSNE(n_components=2).fit_transform(x)
    
    
    x_ = tsne[:, 0]
    y_ = tsne[:, 1]
    plt.scatter(x_, y_, c=y_pred, cmap=plt.cm.get_cmap("jet", 256))
    plt.colorbar(ticks=range(256))
    plt.clim(-0.5, 9.5)
    plt.show()
   
    
    plt.scatter(x_, y_, c=y_pred_kmeans, cmap=plt.cm.get_cmap("jet", 256))
    plt.colorbar(ticks=range(256))
    plt.clim(-0.5, 9.5)
    plt.show()
        
    
    score = silhouette_score (x, y_pred_kmeans, metric='euclidean')
    print ("clusters = {}, Kmeans silhouette score is {})".format(n_clusters, score))
    
    score = silhouette_score (x, y_pred, metric='euclidean')
    print ("For n_clusters = {}, Deep clustering silhouette score is {})".format(n_clusters, score))
    
    for num_clusters in range(2,10):
        clusterer = KMeans(n_clusters=num_clusters, n_jobs=4)
        preds = clusterer.fit_predict(x)
        # centers = clusterer.cluster_centers_
        score = silhouette_score (x, preds, metric='euclidean')
        print ("For n_clusters = {}, Kmeans silhouette score is {})".format(num_clusters, score))
        
    autoencoder = EncoderDecoderBlock(dims,activation,init)
    autoencoder, encoder = autoencoder(loss, optimizer, metrics)
    
    autoencoder.load_weights('dan_1_weights.h5')
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])
    
    if args['plot']=='yes':
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file='model.png', show_shapes=True)
        from IPython.display import Image
        Image(filename='model.png')
        
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(x))
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    y_pred_last = np.copy(y_pred)
    
    model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=optimizer)
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q, _  = model.predict(x, verbose=0)
            p = target_distribution(q)  # update the auxiliary target distribution p
        
            # evaluate the clustering performance
            y_pred = q.argmax(1)
        
            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
        loss = model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

    model.save_weights('b_DANAHER_model_final.h5')
    
    model.load_weights('b_DANAHER_model_final.h5')
    q, _ = model.predict(x, verbose=0)
    p = target_distribution(q)
    y_pred = q.argmax(1)
    score = silhouette_score (x, y_pred, metric='euclidean')
    print ("For n_clusters = {}, Deep clustering silhouette score is {})".format(n_clusters, score))
    
    plt.scatter(x_, y_, c=y_pred, cmap=plt.cm.get_cmap("jet", 256))
    plt.colorbar(ticks=range(256))
    plt.clim(-0.5, 9.5)
    plt.show()
    
    plt.scatter(x_, y_, c=y_pred_kmeans, cmap=plt.cm.get_cmap("jet", 256))
    plt.colorbar(ticks=range(256))
    plt.clim(-0.5, 9.5)
    plt.show()
    
    df['cluster'] = y_pred
    df['cluster'].value_counts()
    df_cluster_0 = df[df['cluster'] == 0]
    df_cluster_0.describe()
    
    
        
    
    
    
    

  
