from matplotlib import pyplot as plt
import numpy as np
import os
from pathvalidate import sanitize_filename
from IPython import display

def plot_acc(
    epoch,
    train_l_list,
    train_acc_list,
    test_acc_list,
    infer_acc_list,
    suptitle=None,
    sub_title=None,
    xlist=None,
    store_pic=True,
    store_path="./rec",
    custom_xlabel=None
):

    if xlist is not None:
        epoch_list = xlist  # Using other variables list
        xlabel=custom_xlabel
    else:
        epoch_list = list(range(1, epoch + 1))
        xlabel="Epoch"

    fig = plt.figure(figsize=(10, 7))

    loss_ylabel="Loss"
    acc_ylabel="Accuracy"

    plt.subplot(2, 2, 1)
    # plt.tight_layout()
    plot_xy(epoch_list, train_l_list, title="Loss",xlabel=xlabel,ylabel=loss_ylabel)
    plt.text(
        epoch_list[-1],
        train_l_list[-1],
        f"{train_l_list[-1]:.6f}",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    plt.subplot(2, 2, 2)
    plot_xy(epoch_list, train_acc_list, title="Train Accuraccy",xlabel=xlabel,ylabel=acc_ylabel)
    plt.text(
        epoch_list[-1],
        train_acc_list[-1],
        f"{train_acc_list[-1]:.6f}",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    plt.subplot(2, 2, 3)
    plot_xy(epoch_list, test_acc_list, title="Test Accuraccy",xlabel=xlabel,ylabel=acc_ylabel)
    plt.text(
        epoch_list[-1],
        test_acc_list[-1],
        f"{test_acc_list[-1]:.6f}",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    try:
        plt.subplot(2, 2, 4)
        plot_xy(epoch_list, infer_acc_list, title="Inference Accuraccy",xlabel=xlabel,ylabel=acc_ylabel)
        plt.text(
            epoch_list[-1],
            infer_acc_list[-1],
            f"{infer_acc_list[-1]:.6f}",
            fontsize=8,
            ha="left",
            va="bottom",
        )
    except ValueError:
        print("No inference data!")
    else:
        pass

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.suptitle(suptitle)
    fig.text(0.5, 0.925, sub_title, fontsize=8, color="blue", ha="center", va="center")
    if store_pic is True:
        os.makedirs(f"{store_path}", exist_ok=True)
        plt.savefig(f"{store_path}/{sanitize_filename(suptitle)}.svg")
    plt.show()


def plot_xy(
    x,
    y,
    xlabel=None,
    ylabel=None,
    title=None,
    save=None,
    figsize=(10, 5),
    show=False,
    x_custom_axis=None,
    sort_y=False,
):

    x_np=np.array(x)
    y_np=np.array(y)

    if x_custom_axis is not None:
        if len(y) != len(x_custom_axis):
            raise ValueError("The length of y and x_custom_axis is not the same."
            "Check your input.")
        
        x_sorted = range(len(x_custom_axis))
        if sort_y is True:
            sorted_indices = np.argsort(y_np)
            x_cus_np=np.array(x_custom_axis)
            y_sorted = y_np[sorted_indices]
            x_cus_sroted= x_cus_np[sorted_indices]
        else:
            y_sorted = y_np
            x_cus_sroted = x_custom_axis
        plt.xticks(x_sorted, x_cus_sroted)
    else:
        # Sort by x-values
        if sort_y is True:
            sorted_indices = np.argsort(y_np)
        else:
            sorted_indices = np.argsort(x_np)
        x_sorted = x_np[sorted_indices]
        y_sorted = y_np[sorted_indices]

    set_figsize(figsize)
    plt.plot(x_sorted, y_sorted)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.title(title, fontsize=12)

    if show is True:
        plt.show()
    if save is not None:
        os.makedirs("pic", exist_ok=True)
        plt.savefig(f"pic/{sanitize_filename(save)}.svg")


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize
