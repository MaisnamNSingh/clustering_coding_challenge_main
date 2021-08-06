import os

from IPython.display import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import plot_model

# need GraphViz to be installed


def plot_save_model(model, to_file):
    plot_model(model, to_file='autoencoder.png', show_shapes=True)
    Image(filename='autoencoder.png')


def plot_save_cluster(embeded_data, prediction, path, to_file):
    tsne_embed = TSNE(n_components=2).fit_transform(embeded_data)
    x = tsne_embed[:, 0]

    y = tsne_embed[:, 1]
    plt.scatter(x, y, c=prediction, cmap=plt.cm.get_cmap("jet", 256))
    plt.colorbar(ticks=range(256))
    plt.clim(-0.5, 9.5)
    plt.show()
    plt.savefig(os.path.join(path, to_file),  bbox_inches='tight')


def plot_save_distribution(data, prediction, path, to_file):
    cols = data.columns
    for col in cols:
        sns_plot = sns.distplot(data[col], kde=False, bins=25)
        sns_plot.savefig(os.path.join(path, to_file))


def plot_save_jointplot(data, path, to_file):
    cols = data.columns
    for col in cols:
        for inner_clog in data.columns:
            sns_plot = sns = sns.jointplot(
                x=col, y=inner_clog, data=data, kind='reg')
            sns_plot.savefig(os.path.join(path, to_file))


def plot_save_distribution(data, path, to_file):
    sns_plot = sns.pairplot(data)
    sns_plot.savefig(os.path.join(path, to_file))


def plot_save_barplot(data, path, to_file):
    cols = data.columns
    for col in cols:
        for inner_col in data.columns:
            sns_plot = sns.barplot(x=col, y=inner_col,
                                   data=data, estimator=np.median)
            sns_plot.savefig(os.path.join(path, to_file))


def plot_save_countplot(data, path, to_file):
    for col in data.columns:
        sns_plot = sns.countplot(x=col, data=data)
        sns_plot.savefig(os.path.join(path, to_file))


def plot_save_heatmap(data, path, to_file, font_size):
    plt.figure(figsize=(8, 6))
    sns.set_context('paper', font_scale=font_size)

    data_heatmap = data.corr()
    sns_plot = sns.heatmap(data_heatmap, annot=True, cmap='Blues')
    sns_plot.savefig(os.path.join(path, to_file))


def plot_save_clustermap(data, path, to_file, font_size):

    plt.figure(figsize=(8, 6))
    sns.set_context('paper', font_scale=font_size)
    sns_plot = sns.clustermap(data, cmap="Blues", standard_scale=1)
    sns_plot.savefig(os.path.join(path, to_file))


def plot_save_facetgrid(data, path, to_file, font_size):
    for col in data.columns:
        sns_plot = sns.FacetGrid(data, col=col, col_wrap=5, height=1.5)
        sns_plot.map(plt.plot, 'solutions', 'score', marker='.')
        sns_plot.savefig(os.path.join(path, to_file))


def plot_save_from_dataframe(data, kind, path, to_file, fontsize):

    if kind == 'hist':
        data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
        plt.show()

    elif kind == 'density':
        data.plot(kind='density', subplots=True, layout=(
            4, 4), sharex=False, legend=False, fontsize=fontsize)
        plt.show()

    elif kind == 'box':
        data.plot(kind='box', subplots=True, layout=(4, 4),
                  sharex=False, sharey=False, legend=False, fontsize=fontsize)
        plt.show()

    plt.savefig(os.path.join(path, to_file),  bbox_inches='tight')
