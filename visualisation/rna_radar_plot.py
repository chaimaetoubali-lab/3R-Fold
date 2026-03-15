import numpy as np
import matplotlib.pyplot as plt


def plot_radar(metrics):

    labels = list(metrics.keys())
    values = list(metrics.values())

    values += values[:1]

    angles = np.linspace(0,2*np.pi,len(labels),endpoint=False)
    angles = np.concatenate([angles,[angles[0]]])

    fig,ax = plt.subplots(subplot_kw={"polar":True})

    ax.plot(angles,values)
    ax.fill(angles,values,alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.title("Model Comparison Radar")

    plt.show()