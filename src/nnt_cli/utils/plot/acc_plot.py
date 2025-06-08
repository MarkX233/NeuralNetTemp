from matplotlib import pyplot as plt
import numpy as np
import os
from pathvalidate import sanitize_filename
from IPython import display
from collections import Counter

def plot_acc(
    epoch,
    train_l_list,
    train_acc_list,
    vali_acc_list,
    infer_acc_list,
    suptitle=None,
    sub_title=None,
    xlist=None,
    store_pic=True,
    store_path="./rec",
    custom_xlabel=None,
    text_in_sub_4=None,
    cus_title=['Loss', 'Train Accuracy', 'Validation Accuracy', 'Inference Accuracy'],
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
    plot_xy(epoch_list, train_l_list, title=cus_title[0],xlabel=xlabel,ylabel=loss_ylabel)
    plt.text(
        epoch_list[-1],
        train_l_list[-1],
        f"{train_l_list[-1]:.6f}",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    plt.subplot(2, 2, 2)
    plot_xy(epoch_list, train_acc_list, title=cus_title[1],xlabel=xlabel,ylabel=acc_ylabel)
    plt.text(
        epoch_list[-1],
        train_acc_list[-1],
        f"{train_acc_list[-1]:.6f}",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    plt.subplot(2, 2, 3)
    plot_xy(epoch_list, vali_acc_list, title=cus_title[2],xlabel=xlabel,ylabel=acc_ylabel)
    plt.text(
        epoch_list[-1],
        vali_acc_list[-1],
        f"{vali_acc_list[-1]:.6f}",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    try:
        if text_in_sub_4 is None:
            plt.subplot(2, 2, 4)
            plot_xy(epoch_list, infer_acc_list, title=cus_title[3],xlabel=xlabel,ylabel=acc_ylabel)
            plt.text(
                epoch_list[-1],
                infer_acc_list[-1],
                f"{infer_acc_list[-1]:.6f}",
                fontsize=8,
                ha="left",
                va="bottom",
            )
        else:
            plt.subplot(2, 2, 4)
            plt.axis('off')
            text_x = 0.5
            text_y = 0.5
            text_content = "\n".join(text_in_sub_4)
            plt.text(
                text_x, text_y,
                text_content,
                fontsize=10,
                ha='center',
                va='center',
                # bbox=dict(
                #     boxstyle='round,pad=0.5',
                #     facecolor='whitesmoke',
                #     edgecolor='lightgray',
                #     alpha=0.8
                # ),
                transform=plt.gca().transAxes
            )
            plt.title(cus_title[3], fontsize=12, pad=10)
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
    mark_points=None,
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

    # mp_np = np.array(mark_points) if isinstance(mark_points, list) else mark_points
    # mp_sorted = mp_np[sorted_indices] if isinstance(mark_points, np.ndarray) else mp_np

    if isinstance(mark_points, list):
        mp_sorted = np.array(mark_points)[sorted_indices]
    elif isinstance(mark_points, np.ndarray):
        mp_sorted = mark_points[sorted_indices]
    else:
        mp_sorted = mark_points

    mark_points_on_plot(mp_sorted, x_sorted, y_sorted, plot_func)
    

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
    mark_points_y=None,
    mark_points_z=None,
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

    # mp_np_y = np.array(mark_points_y) if isinstance(mark_points_y, list) else mark_points_y
    # mp_sorted_y = mp_np_y[sorted_indices] if isinstance(mark_points_y, np.ndarray) else mp_np_y

    if isinstance(mark_points_y, list):
        mp_sorted_y = np.array(mark_points_y)[sorted_indices]
    elif isinstance(mark_points_y, np.ndarray):
        mp_sorted_y = mark_points_y[sorted_indices]
    else:
        mp_sorted_y = mark_points_y

    mark_points_on_plot(mp_sorted_y, x_sorted, y_sorted, plot_func_y, color=y_color)

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
        
        # mp_np_z = np.array(mark_points_z) if isinstance(mark_points_z, list) else mark_points_z
        # mp_sorted_z = mp_np_z[sorted_indices] if isinstance(mark_points_z, np.ndarray) else mp_np_z

        if isinstance(mark_points_z, list):
            mp_sorted_z = np.array(mark_points_z)[sorted_indices]
        elif isinstance(mark_points_z, np.ndarray):
            mp_sorted_z = mark_points_z[sorted_indices]
        else:
            mp_sorted_z = mark_points_z

        mark_points_on_plot(mp_sorted_z, x_sorted, y_sorted, plot_func_z, color=z_color)

    ax1.set_xticks(x_sorted)
    ax1.set_xticklabels(x_cus_sroted, rotation=45,ha='right')    
    # plt.xticks(x_sorted, x_cus_sroted, rotation=45)        
    plt.title(title, fontsize=12)
    

    if save is not None:
        os.makedirs("pic", exist_ok=True)
        plt.savefig(f"pic/{sanitize_filename(save)}.svg")

    if show is True:
        plt.show()
    
from matplotlib.gridspec import GridSpec
def plot_broken_xy(
    x,
    y,
    bx_indices=None,
    by_indices=None,
    xlabel=None,
    ylabel=None,
    title=None,
    save=None,
    figsize=(10, 5),
    show=False,
    plot_func=plt.scatter,
    mark_points=None,
    # mp_bx_indices=None,
    # mp_by_indices=None,
    addjust_marks=False,
):
    """
    Scatter plot for x and y with broken axes.
    
    """
    
    c_x_main = Counter(x)
    c_y_main = Counter(y)

    if bx_indices is not None:
        x_bx = [x[i] for i in bx_indices]
        y_bx = [y[i] for i in bx_indices]
        # x_main = [x for i, x in enumerate(x_main) if i not in bx_indices]
        # y_main = [y for i, y in enumerate( y_main) if i not in bx_indices]
        c_x_main = c_x_main - Counter(x_bx)
        c_y_main = c_y_main - Counter(y_bx)

    
    if by_indices is not None:
        x_by = [x[i] for i in by_indices]
        y_by = [y[i] for i in by_indices]
        # x_main = [x for i, x in enumerate(x_main) if i not in by_indices]
        # y_main = [y for i, y in enumerate( y_main) if i not in by_indices]
        c_x_main = c_x_main - Counter(x_by)
        c_y_main = c_y_main - Counter(y_by)
    
    x_main = list(c_x_main.elements())
    y_main = list(c_y_main.elements())

    if mark_points is not None:
        if isinstance(mark_points, list):
            c_mp_m = Counter(mark_points)
            if bx_indices is not None:
                mark_points_bx = [mp for i, mp in enumerate(mark_points) if i in bx_indices]
                # mark_points_m = [mp for i, mp in enumerate(mark_points) if i not in bx_indices]
                c_mp_m = c_mp_m - Counter(mark_points_bx)
                
            
            if by_indices is not None:
                mark_points_by = [mp for i, mp in enumerate(mark_points) if i in by_indices]
                # mark_points_m = [mp for i, mp in enumerate(mark_points_m) if i not in by_indices]
                c_mp_m = c_mp_m - Counter(mark_points_by)
            
            mark_points_m = list(c_mp_m.elements())
        
        else:
            mark_points_m = mark_points

    fig = plt.figure(figsize=figsize)


    gs = GridSpec(
                    2 if by_indices is not None else 1, 
                    2 if bx_indices is not None else 1, 
                    width_ratios=[1, 0.2] if bx_indices is not None else None, 
                    height_ratios=[1,0.2] if by_indices is not None else None, 
                    wspace=0.05, 
                    hspace=0.05
                )

    if xlabel:
        fig.supxlabel(xlabel, fontsize=10)
    if ylabel:
        fig.supylabel(ylabel, fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12)

    ax1 = fig.add_subplot(gs[0,0])
    # ax1.grid(True, linestyle='--', alpha=0.7)
    plt.sca(ax1)
    plot_func(x_main, y_main)


    # ax1.spines['right'].set_visible(False)
    
    if addjust_marks is True:
        annotate_text(ax1, x_main, y_main, mark_points_m)
    else:
        mark_points_on_plot(mark_points_m, x_main, y_main, plot_func)

    if bx_indices is not None:
        ax2 = fig.add_subplot(gs[0,1], sharey=ax1)
        # ax2.grid(True, linestyle='--', alpha=0.7)
        plt.sca(ax2)
        plot_func(x_bx, y_bx)
        # ax2.set_xlabel(f"{xlabel}(Outlier)", fontsize=10)
        plt.setp(ax2.get_yticklabels(), visible=False)

        # ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(True)
        ax2.yaxis.set_ticks_position('right') # Move the scale to the right

        if addjust_marks is True:
            annotate_text(ax2, x_bx, y_bx, mark_points_bx)
        else:
            mark_points_on_plot(mark_points_bx, x_bx, y_bx, plot_func)

        
    if by_indices is not None:
        ax3 = fig.add_subplot(gs[1,0], sharex=ax1)
        # ax3.grid(True, linestyle='--', alpha=0.7)
        plt.sca(ax3)
        plot_func(x_by, y_by)
        # ax3.set_xlabel(f"{xlabel}(Outlier)", fontsize=10)
        plt.setp(ax3.get_xticklabels(), visible=True)

        # ax3.spines['Top'].set_visible(False)
        # ax3.spines['right'].set_visible(True)
        # ax3.yaxis.set_ticks_position('right')

        if addjust_marks is True:
            annotate_text(ax3, x_by, y_by, mark_points_by)
        else:
            mark_points_on_plot(mark_points_by, x_by, y_by, plot_func)

    # add_broken_marks(ax1, ax2)   


    if save is not None:
        os.makedirs("pic", exist_ok=True)
        plt.savefig(f"pic/{sanitize_filename(save)}.svg")
    
    if show is True:
        plt.show()

from brokenaxes import brokenaxes
# Note: brokenaxes is a third-party library, you need to install it first.
def plot_broken(
    x_main,
    y_main,
    x_sec,
    y_sec,
    xlabel=None,
    ylabel=None,
    title=None,
    save=None,
    figsize=(10, 5),
    show=False,
    plot_func_str='plot',
    mark_points_m=None,
    mark_points_s=None,
    addjust_marks=False,
    sharex=False,
    sharey=False,
):
    
    """
    Plot two sets of data on broken axes.
    Currently, annotating text is not working well with broken axes.
    """
    PAD_SCALE = 0.1  # Padding for the broken axes
    min_x_m = min(x_main)
    max_x_m = max(x_main)
    min_x_s = min(x_sec)
    max_x_s = max(x_sec)

    x_len_m = max_x_m - min_x_m
    x_len_s = max_x_s - min_x_s

    if x_len_m == 0:
        x_len_m = 10
    if x_len_s == 0:
        x_len_s = 10
    
    x_lim_m = (min_x_m - (x_len_m) * PAD_SCALE, 
               max_x_m + (x_len_m) * PAD_SCALE)
    x_lim_s = (min_x_s - (x_len_s) * PAD_SCALE, 
               max_x_s + (x_len_s) * PAD_SCALE)
    
    x=[*x_main, *x_sec]
    y=[*y_main, *y_sec]

    if sharey is True:
        min_y_m = min(y)
        max_y_m = max(y)
        min_y_s = min(y)
        max_y_s = max(y)
    else:
        min_y_m = min(y_main)
        max_y_m = max(y_main)
        min_y_s = min(y_sec)
        max_y_s = max(y_sec)
    
    y_len_m = max_y_m - min_y_m
    y_len_s = max_y_s - min_y_s

    if y_len_m == 0:
        y_len_m = 10
    if y_len_s == 0:
        y_len_s = 10
    
    y_lim_m = (min_y_m - (y_len_m) * PAD_SCALE,
               max_y_m + (y_len_m) * PAD_SCALE)
    y_lim_s = (min_y_s - (y_len_s) * PAD_SCALE,
               max_y_s + (y_len_s) * PAD_SCALE)
    
    fig = plt.figure(figsize=figsize)
    bax = brokenaxes(
                    xlims=None if sharex else (x_lim_m, x_lim_s),
                    ylims=None if sharey else (y_lim_m, y_lim_s),
                    hspace=0.1,  # Adjust the space between the two axes
                    wspace=0.1,
                    despine=False,
    )

    if isinstance(plot_func_str, str):
        if hasattr(bax, plot_func_str):
            plot_method = getattr(bax, plot_func_str)
        else:
            raise ValueError(f"ax has no method '{plot_func_str}'")
    else:
        raise ValueError("plot_func must be a string representing a valid matplotlib method, e.g., 'plot', 'bar', etc.")
    
    # plt.sca(bax)
    # plot_func(x, y)
    plot_method(x, y)
    
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.title(title, fontsize=12)
    
    mp= [*mark_points_m, *mark_points_s]
    if addjust_marks is True:
        annotate_text(bax, x, y, mp)
    else:
        mark_points_on_plot(mp, x, y, plot_func_str)


    if save is not None:
        os.makedirs("pic", exist_ok=True)
        plt.savefig(f"pic/{sanitize_filename(save)}.svg")
    
    if show is True:
        plt.show()


def mark_points_on_plot(mark_points, x, y, plot_func, color='black'):

    if isinstance(mark_points, list) or isinstance(mark_points, np.ndarray):
        for i, txt in enumerate(mark_points):
            plt.annotate(
                txt,
                (x[i], y[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
                color=color,
            )
    elif mark_points == 'True' or mark_points is True:
        if plot_func == plt.bar or plot_func == 'bar':   
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
                    color=color,
                )
        else:
            for i, txt in enumerate(y):
                plt.annotate(
                    f"{txt:.2f}",
                    (x[i], y[i]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=10,
                    color=color,
                    )
                


from adjustText import adjust_text
def annotate_text(ax, x, y, texts):
    text_anno = []

    if isinstance(texts, str):
        text_anno.append(ax.text(x, y, texts,
                          fontsize=9, alpha=0.8, ha='right'))
    elif isinstance(texts, list) or isinstance(texts, np.ndarray):
        for i, txt in enumerate(texts):
            text_anno.append(ax.text(x[i], y[i], txt,
                            fontsize=9, alpha=0.8, ha='right'))
    adjust_text(text_anno, arrowprops=dict(arrowstyle="-", color='black',  lw=0.7, alpha=0.3),)
