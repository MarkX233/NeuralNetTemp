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
    plot_func=plt.plot,
    mark_points=False,
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
        plt.xticks(x_sorted, x_cus_sroted, rotation=45)
    else:
        # Sort by x-values
        if sort_y is True:
            sorted_indices = np.argsort(y_np)
        else:
            sorted_indices = np.argsort(x_np)
        x_sorted = x_np[sorted_indices]
        y_sorted = y_np[sorted_indices]

    set_figsize(figsize)
    plot_func(x_sorted, y_sorted)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.title(title, fontsize=12)

    if mark_points is True:   
        for p in plt.gca().patches:
            height = p.get_height()
            width = p.get_width()
            plt.text(
                p.get_x() + width / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    if save is not None:
        os.makedirs("pic", exist_ok=True)
        plt.savefig(f"pic/{sanitize_filename(save)}.svg")

    if show is True:
        plt.show()
    


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    # use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


def plot_double(
    x,
    y,
    z=None,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    title=None,
    save=None,
    figsize=(10, 5),
    show=False,
    x_custom_axis=None,
    sort_y=False,
    plot_func_y=plt.plot,
    plot_func_z=plt.plot,
    mark_points=False,
    y_color="#167496",
    z_color="#BF4D19",
    z_scale_factor=1.0,
):

    x_np=np.array(x)
    y_np=np.array(y)


    if x_custom_axis is not None:
        if len(y) != len(x_custom_axis):
            raise ValueError("The length of y and x_custom_axis is not the same."
            "Check your input.")
        if z is not None and len(z) != len(x_custom_axis):
            raise ValueError("The length of z and x_custom_axis is not the same."
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
    else:
        if len(y) != len(x_np):
            raise ValueError("The length of y and x is not the same."
            "Check your input.")
        if z is not None:
            if len(z) != len(x_np):
                raise ValueError("The length of z and x is not the same."
                "Check your input.")
            
        # Default: Sort by x-values
        if sort_y is True:
            sorted_indices = np.argsort(y_np)
        else:
            sorted_indices = np.argsort(x_np)
        x_sorted = x_np[sorted_indices]
        y_sorted = y_np[sorted_indices]
    
    if z is not None:
        z_np = np.array(z)
        z_sorted = z_np[sorted_indices]

    fig, ax1 = plt.subplots(figsize=figsize)
    plt.sca(ax1)
    plot_func_y(x_sorted, y_sorted, color=y_color)
    ax1.set_xlabel(xlabel, fontsize=10)
    ax1.set_ylabel(ylabel, fontsize=10, color=y_color)
    ax1.tick_params(axis='y', labelcolor=y_color)

    if mark_points is True:   
        for p in ax1.patches:
            height = p.get_height()
            width = p.get_width()
            ax1.text(
                p.get_x() + width / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=y_color
            )

    if z is not None:
        ax2 = ax1.twinx()
        plt.sca(ax2)
        plot_func_z(x_sorted, z_sorted, color=z_color)	
        z_min, z_max = np.min(z_sorted), np.max(z_sorted)
        scaled_zmax = z_max * z_scale_factor
        # ax2.set_ylim(z_min, scaled_zmax)
        ax2.set_ylim(top=scaled_zmax)
        ax2.set_ylabel(zlabel, fontsize=10, color=z_color)
        ax2.tick_params(axis='y', labelcolor=z_color)
        
        if mark_points:
            for p in ax2.patches:
                height = p.get_height()
                width = p.get_width()
                ax2.text(p.get_x() + width/2, height, 
                         f"{height:.2f}", ha='center', va='bottom', 
                         fontsize=10, color=z_color)
                
    plt.title(title, fontsize=12)
    plt.xticks(x_sorted, x_cus_sroted, rotation=45)

    if save is not None:
        os.makedirs("pic", exist_ok=True)
        plt.savefig(f"pic/{sanitize_filename(save)}.svg")

    if show is True:
        plt.show()
    
