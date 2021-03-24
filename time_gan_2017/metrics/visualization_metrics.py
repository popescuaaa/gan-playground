import plotly.graph_objects as go
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


def visualize(generated_data: np.ndarray, real_data: np.ndarray):

    # Do t-SNE Analysis together
    processed_data = np.concatenate((real_data, generated_data), axis=0)
    t_sne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    t_sne_results = t_sne.fit_transform(processed_data)

    samples_number = real_data.shape[0]
    colors = ["red" for _ in range(samples_number)] + ["blue" for _ in range(samples_number)]

    fig = go.Figure(data=go.Scatter(x=t_sne_results[:samples_number], y=t_sne_results[samples_number:], mode='markers'))
    return fig
