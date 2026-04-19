import matplotlib.pyplot as plt
import numpy as np
def lossPlot(losses,title="", save_as="../visualizations/pic.png",ax=None,):
    #plot indivisual lines on the graph
    for loss in losses:
        values = loss["values"]
        label = loss["label"]
        color = loss["color"]
        
        if ax==None:
            plt.plot(values, label=label, color=color)
        else:
            ax.plot(values, label=label, color=color)
        
    #visual esthetics added to the plot
    if ax==None:
        plt.xlabel("Epoch")
        plt.ylabel("Loss MAE")
        plt.ylim(0, 20)
        plt.yticks(np.arange(0, 20, 1))
        plt.title(title)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.savefig(save_as)
        plt.show()
    else:
        ax.set_ylabel("Loss MAE")
        ax.set_ylim(0, 20)
        ax.set_yticks(np.arange(0, n, 1))
        ax.set_title(title)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        ax.legend()