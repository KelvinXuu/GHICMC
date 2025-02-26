import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def loss_plot(loss, acc, nmi, ari, dataset):
    epochs = np.array(range(1, len(loss) + 1))
    fig = plt.figure(figsize=(8, 5))
    ax_left = fig.add_subplot(111)
    ax_right = ax_left.twinx()

    ax_left.set_xlabel('Epoch', fontsize=14)
    ax_left.set_ylabel('Clustering Performance', fontsize=14)
    ax_right.set_ylabel('Loss', fontsize=14)

    a1 = ax_right.plot(epochs, loss, color='#F27970', label='Loss')
    a2 = ax_left.plot(epochs, acc, color='#54B345', label='ACC')
    a3 = ax_left.plot(epochs, nmi, color='#05B9E2', label='NMI')
    a4 = ax_left.plot(epochs, ari, color='#BB9727', label='ARI')

    lns = a1 + a2 + a3 + a4
    labs = [l.get_label() for l in lns]
    ax_left.legend(lns, labs, loc='center right')

    plt.tight_layout()
    plt.savefig(f'../img/{dataset}_loss.pdf')
    plt.savefig(f'../img/{dataset}_loss.jpg')
    plt.close()
