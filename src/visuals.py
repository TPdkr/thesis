import gc
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
from types import SimpleNamespace

COLORS = SimpleNamespace(
    YOLO=SimpleNamespace(train="deepskyblue", test="navy"),
    DINO=SimpleNamespace(train="orangered",  test="maroon"),
    DINOP=SimpleNamespace(train="gold", test="darkgoldenrod"),
    VGG=SimpleNamespace(train="cyan",        test="darkolivegreen"),
    DE=SimpleNamespace(train="navy",         test="indigo"),
)

def lossPlot(losses,title="", save_as="../visualizations/pic.png",ax=None,):
    """
    Create a simple loss plot between several arrays

    Args:
        losses: dictionary list containing the data to be plotted
        title: the title of the plot
        save_as: save the plot as a file to this location
        ax: if provided axes is used instead of big plot
    """
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
        ax.set_yticks(np.arange(0, 20, 1))
        ax.set_title(title)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        ax.legend()

def overviewPlot(error_scores_df, depths_train_df, depths_test_df, title, save_as="../visualizations/pic.png", markers=True, train_color="royalblue", test_color="darkred"):
    # TRAINING AND TEST LOS IN EACH EPOCH
    train_error = error_scores_df["train_error"].tolist()
    test_error = error_scores_df["test_error"].tolist()

    # ERROR RATE DEPENDING ON DEPTH
    error_fn = nn.L1Loss()
    error= np.array([])
    for i in range (8):
        # masks
        depths_in_range_true = (depths_test_df["true_depths_test"]<=(i+1)*10) & (depths_test_df["true_depths_test"]>=i*10)

        # data 
        true_depths = depths_test_df["true_depths_test"][depths_in_range_true]
        predicted_depths = depths_test_df["predicted_depths_test"][depths_in_range_true]

        error_rate = error_fn(torch.from_numpy(true_depths.values), torch.from_numpy(predicted_depths.values))
        error = np.append(error, error_rate.item())

    # TOTAL ERROR RATES

    MAE = error_fn(
        torch.from_numpy(depths_test_df["true_depths_test"].values),
        torch.from_numpy(depths_test_df["predicted_depths_test"].values)
    ).item()

    error_fn_rmse = torch.nn.MSELoss()
    RMSE = error_fn_rmse(
        torch.from_numpy(depths_test_df["true_depths_test"].values),
        torch.from_numpy(depths_test_df["predicted_depths_test"].values)
    ).item()

    title_suffix = f" MAE: {MAE:.3f} RMSE: {RMSE:.3f}"

    # BASIC PLOT OBJECT CREATED
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    # PLOT THE DATA
    for i, ax in enumerate(axes):
        if i == 0:
            if markers:
                ax.plot(train_error, label="Train Loss", color=train_color, marker='o')
                ax.plot(test_error, label="Validation Loss", color=test_color, marker='s')
            else: 
                ax.plot(train_error, label="Train Loss", color=train_color)
                ax.plot(test_error, label="Validation Loss", color=test_color)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss MAE")
            ax.set_ylim(0, 15)
            ax.set_xlim(0, len(train_error)-1)
            ax.set_yticks(np.arange(0, 15, 1))
            ax.set_title("Training Loss over Time")
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
            ax.legend()
        if i==1:
            #true depths are shown along the predicted depths for tain
            ax.hist(depths_train_df["true_depths"], bins=30, label="True Depths Train", color=train_color, alpha=0.7)
            ax.hist(depths_train_df["predicted_depths"], bins=30, label="Predicted Depths Train", color=test_color, alpha=0.7)
            ax.set_xlabel("True Depths")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of True Depths in training")
            ax.legend()
        if i==2:
            #true depths are shown along the predicted depths for test
            ax.hist(depths_test_df["true_depths_test"], bins=30, label="True Depths Test", color=train_color, alpha=0.7)
            ax.hist(depths_test_df["predicted_depths_test"], bins=30, label="Predicted Depths Test", color=test_color, alpha=0.7)
            ax.set_xlabel("True Depths")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of True Depths in testing")
            ax.legend()
        if i==3:
            ax.plot(np.arange(0, 80, 10), error, label="Error Rate", color=test_color, marker='o')
            ax.set_xlabel("Depth Range (m)")
            ax.set_ylabel("Loss MAE")
            ax.set_title("Error Rate vs depth")
            ax.set_xticks(np.arange(0, 80, 10))
            ax.set_ylim(ymin=0, ymax=10)
            ax.set_yticks(np.arange(0,10,1))
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
            ax.legend()

    plt.suptitle(f"{title}{title_suffix}")
    plt.tight_layout()
    fig.savefig(save_as)
    plt.show()

    del error_scores_df, depths_train_df, depths_test_df, error
    gc.collect()