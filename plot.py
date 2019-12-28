# plot graph topology animation
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation



def simple_update(t, layout, G_list, ax):
    ax.clear()

    nx.draw(G_list[t], pos=layout, ax=ax, with_labels = True)

    # Set the title
    ax.set_title("Frame {}".format(t))

def plot_topo_dynamic(G_list,path = 'animation.gif'):
    # Build plot
    fig, ax = plt.subplots(figsize=(6,4))
    layout = nx.circular_layout(G_list[0])
    ani = animation.FuncAnimation(fig, simple_update, frames=len(G_list), fargs=(layout, G_list, ax))
    ani.save(path, writer='imagemagick')

    # plt.show()
