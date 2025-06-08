from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import numpy as np
from tqdm import tqdm


def visualize_distribution_robust(data_matrix, title="Robust Distribution Analysis", quantile=0.99, font_scale=1.0):
    """
    Visualize data distribution with outlier handling through quantile-based scaling
    
    Parameters:
        data_matrix (np.ndarray or Tensor): Input data in [Time_steps, Channels] format
        title (str): Visualization title
        quantile (float): Quantile threshold for main distribution (0-1)
    """
    # Convert tensor to numpy if needed
    if isinstance(data_matrix, torch.Tensor):
        data_matrix = data_matrix.numpy()


    if data_matrix.ndim > 2:
        data_matrix = data_matrix.reshape(data_matrix.shape[0], -1)

    original_rc = {
        'font.size': get_rc_float('font.size'),
        'axes.titlesize': get_rc_float('axes.titlesize'),
        'axes.labelsize': get_rc_float('axes.labelsize'),
        'xtick.labelsize': get_rc_float('xtick.labelsize'),
        'ytick.labelsize': get_rc_float('ytick.labelsize'),
        'figure.titlesize': get_rc_float('figure.titlesize')
    }
    
    with plt.rc_context({
        'font.size': original_rc['font.size'] * font_scale,
        'axes.titlesize': original_rc['axes.titlesize'] * font_scale,
        'axes.labelsize': original_rc['axes.labelsize'] * font_scale,
        'xtick.labelsize': original_rc['xtick.labelsize'] * font_scale,
        'ytick.labelsize': original_rc['ytick.labelsize'] * font_scale,
        'figure.titlesize': original_rc['figure.titlesize'] * font_scale
    }):
        # Calculate distribution metrics
        flat_data = data_matrix.flatten()
        data_min = np.min(flat_data)
        data_max = np.max(flat_data)
        q_value = np.quantile(flat_data, quantile)
        main_data_mask = (flat_data >= data_min) & (flat_data <= q_value)
        outlier_mask = ~main_data_mask

        # Create figure layout
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig)
        fig.suptitle(f"{title}\n(Quantile Threshold: {quantile}, Cutoff Value: {q_value:.2e})", 
                    fontsize=14 * font_scale, y=0.95)

        # 1. Main Distribution Histogram with Inset
        ax_hist = fig.add_subplot(gs[0, 0])
        ax_hist_inset = ax_hist.inset_axes([0.55, 0.55, 0.4, 0.4])
        
        # Main histogram (linear scale)
        bins_main = np.linspace(data_min, q_value, 50)
        ax_hist.hist(flat_data[main_data_mask], bins=bins_main, 
                    color='tab:blue', edgecolor='white')
        ax_hist.set_title('Core Distribution (Linear Scale)')
        ax_hist.set_xlabel('Value')
        ax_hist.set_ylabel('Frequency')
        
        # Inset histogram (dynamic scale)
        if data_min > 0:
            # Use log scale for positive data
            bins_inset = np.geomspace(data_min, data_max, 50)
            ax_hist_inset.hist(flat_data, bins=bins_inset, color='tab:orange', log=True)
            ax_hist_inset.set_yscale('log')
        else:
            # Use linear scale for data with negative values
            bins_inset = np.linspace(data_min, data_max, 50)
            ax_hist_inset.hist(flat_data, bins=bins_inset, color='tab:orange')
        ax_hist_inset.axvline(q_value, color='red', linestyle='--', linewidth=1)
        ax_hist_inset.set_title('Full Range Histogram', fontsize=8 * font_scale)
        ax_hist_inset.tick_params(labelsize=6)

        # 2. Temporal Distribution with Outlier Highlight
        ax_time = fig.add_subplot(gs[0, 1:])
        time_coords = np.repeat(np.arange(data_matrix.shape[0]), data_matrix.shape[1])
        
        # Plot main distribution
        time_scatter = ax_time.scatter(
            x=time_coords[main_data_mask],
            y=flat_data[main_data_mask],
            c=np.tile(np.arange(data_matrix.shape[1]), data_matrix.shape[0])[main_data_mask],
            cmap='viridis',
            alpha=0.5,
            s=10,
            edgecolors='none',
            vmin=0,
            vmax=data_matrix.shape[1]
        )
        
        # Highlight outliers
        if np.any(outlier_mask):
            ax_time.scatter(
                x=time_coords[outlier_mask],
                y=flat_data[outlier_mask],
                c='red',
                s=30,
                marker='x',
                label=f'Outliers (< {data_min:.2e} or > {q_value:.2e})'
            )
            ax_time.legend()
        
        ax_time.set_title('Temporal Distribution')
        ax_time.set_xlabel('Time Step')
        ax_time.set_ylabel('Value')
        plt.colorbar(time_scatter, ax=ax_time, label='Channel ID')

        # 3. Channel-wise Analysis with Statistics Table
        ax_channels = fig.add_subplot(gs[1:, :2])
        
        # Calculate statistics
        main_data = data_matrix[(data_matrix >= data_min) & (data_matrix <= q_value)]
        global_mean = np.nanmean(main_data)
        global_std = np.nanstd(main_data)
        channel_means = np.array([np.nanmean(data_matrix[:,c][(data_matrix[:,c] >= data_min) & 
                                                            (data_matrix[:,c] <= q_value)]) 
                                for c in range(data_matrix.shape[1])])
        channel_stds = np.array([np.nanstd(data_matrix[:,c][(data_matrix[:,c] >= data_min) & 
                                                        (data_matrix[:,c] <= q_value)]) 
                            for c in range(data_matrix.shape[1])])

        channel_stats = {
            'mean': np.mean(data_matrix, axis=0),
            'q75': np.quantile(data_matrix, 0.75, axis=0),
            'quantile': np.quantile(data_matrix, quantile, axis=0),
            'max': np.max(data_matrix, axis=0)
        }
        
        x = np.arange(data_matrix.shape[1])
        ax_channels.fill_between(x, channel_stats['q75'], channel_stats['quantile'],
                                color='orange', alpha=0.3, label='75-99% Range')
        ax_channels.plot(x, channel_stats['mean'], color='blue', label='Mean')
        ax_channels.plot(x, channel_stats['quantile'], color='red', linestyle='--', 
                        label=f'{quantile*100:.0f}% Quantile')
        
        # Annotate outlier channels
        outlier_channels = np.where(channel_stats['max'] > q_value)[0]
        for ch in outlier_channels:
            ax_channels.annotate(f'Ch{ch}', (ch, channel_stats['quantile'][ch]),
                                textcoords="offset points",
                                xytext=(0,5), ha='center', fontsize=8 * font_scale,
                                color='darkred')


        # Add statistics table
        stats_table_data = [
            [f"Metric (< {quantile*100:.0f}%)", "Value"],
            ["Global Mean", f"{global_mean:.2e}"],
            ["Global Std", f"{global_std:.2e}"],
            ["Chan Mean Avg", f"{np.nanmean(channel_means):.2e}"],
            ["Chan Std Avg", f"{np.nanmean(channel_stds):.2e}"]
        ]
        
        table = ax_channels.table(
            cellText=stats_table_data,
            colWidths=[0.25, 0.25],
            loc='upper right',
            bbox=[0.72, 0.6, 0.25, 0.3],  # Table [x, y, width, height]
            edges='closed'
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('gray')
            if row == 0:
                cell.set_facecolor('#F0F0F0')
                cell.set_text_props(weight='bold')

        ax_channels.set_title('Channel-wise Distribution Analysis')
        ax_channels.set_xlabel('Channel ID')
        ax_channels.set_ylabel('Value')
        ax_channels.legend(ncol=2, fontsize=9 * font_scale, loc='upper right')
        ax_channels.grid(True, alpha=0.3)
        ax_channels.set_xticks(np.linspace(0, data_matrix.shape[1], 10, dtype=int))

        # 4. Value-Channel Density Heatmap
        ax_density = fig.add_subplot(gs[1:, 2])
        
        valid_values = flat_data[main_data_mask]
        data_min = np.min(valid_values)
        data_max = np.max(valid_values)


        if data_max - data_min < 1e-6:
            bins = [data_min, data_max+1e-6]
        elif data_min <= 0 or (data_max - data_min) / data_max < 0.01:
            bins = np.linspace(data_min, data_max, 50)
        else: 
            bins = np.geomspace(max(1e-6, data_min), data_max, 50)

        hb = ax_density.hexbin(
            x=valid_values,
            y=np.tile(np.arange(data_matrix.shape[1]), data_matrix.shape[0])[main_data_mask],
            gridsize=(50, data_matrix.shape[1]//2),
            cmap='coolwarm' if data_min < 0 else 'plasma',
            bins=bins,
            mincnt=1,
            extent=(data_min, data_max, 0, data_matrix.shape[1])
        )
        ax_density.set_title('Value-Channel Density (Core Distribution)')
        ax_density.set_xlabel('Value')
        ax_density.set_ylabel('Channel ID')
        plt.colorbar(hb, ax=ax_density, label='Density')

        plt.tight_layout()
        plt.show()
    

def visualize_value_distribution(event_matrix, title="Value Distribution Analysis", font_scale=1.0):
    """
    Visualize event value distribution across multiple dimensions for normalization
    Suitable for data with negative values.
    """
    if isinstance(event_matrix, torch.Tensor):
        event_matrix = event_matrix.numpy()


    original_rc = {
        'font.size': get_rc_float('font.size'),
        'axes.titlesize': get_rc_float('axes.titlesize'),
        'axes.labelsize': get_rc_float('axes.labelsize'),
        'xtick.labelsize': get_rc_float('xtick.labelsize'),
        'ytick.labelsize': get_rc_float('ytick.labelsize'),
        'figure.titlesize': get_rc_float('figure.titlesize')
    }
    
    with plt.rc_context({
        'font.size': original_rc['font.size'] * font_scale,
        'axes.titlesize': original_rc['axes.titlesize'] * font_scale,
        'axes.labelsize': original_rc['axes.labelsize'] * font_scale,
        'xtick.labelsize': original_rc['xtick.labelsize'] * font_scale,
        'ytick.labelsize': original_rc['ytick.labelsize'] * font_scale,
        'figure.titlesize': original_rc['figure.titlesize'] * font_scale
    }):

        # Reshape and prepare data
        T = event_matrix.shape[0]
        if event_matrix.ndim > 2:
            # C = np.prod(event_matrix.shape[1:-1])
            event_matrix = event_matrix.reshape(T, -1)
        
        C = event_matrix.shape[1]

        flat_values = event_matrix.flatten()
        time_coords = np.repeat(np.arange(T), C)
        channel_coords = np.tile(np.arange(C), T)
        
        # Calculate data properties
        data_min = np.min(flat_values)
        data_max = np.max(flat_values)
        has_negative = data_min < 0

        # Create figure layout
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig)
        fig.suptitle(title, fontsize=14 * font_scale)

        # 1. Adaptive Value Histogram
        ax_hist = fig.add_subplot(gs[0, 0])
        
        # Dynamic binning
        if has_negative:
            bins = np.linspace(data_min, data_max, 50)
            yscale = 'linear'
        else:
            bins = np.geomspace(max(1e-6, data_min), data_max, 50)
            # bins = np.logspace(max(1e-6, data_min), data_max, 50)
            yscale = 'log'
        
        counts, bins, _ = ax_hist.hist(flat_values, bins=bins, 
                                    color='skyblue', edgecolor='navy')
        
        # Add percentiles
        percentiles = np.percentile(flat_values, [25, 50, 75, 95, 99])
        for p in percentiles:
            ax_hist.axvline(p, color='red', linestyle='--', alpha=0.7)
        
        ax_hist.set_title(f'Value Distribution Histogram ({yscale} Scale)')
        ax_hist.set_xlabel('Values')
        ax_hist.set_ylabel('Count')
        ax_hist.set_yscale(yscale)

        # 2. Temporal Heatmap
        ax_time = fig.add_subplot(gs[0, 1:])
        cmap = 'coolwarm' if has_negative else 'viridis'
        sc = ax_time.scatter(time_coords, flat_values, c=channel_coords,
                            cmap=cmap, alpha=0.3, s=5, 
                            vmin=0 if has_negative else None)
        plt.colorbar(sc, ax=ax_time, label='Channel ID')
        ax_time.set_title('Temporal Distribution')
        ax_time.set_xlabel('Time Steps')
        ax_time.set_ylabel('Values')

        # 3. Channel Violin Plot
        ax_violin = fig.add_subplot(gs[1:, :2])
        vparts = ax_violin.violinplot(
            [event_matrix[:,c] for c in range(C)], 
            showmeans=True, 
            showextrema=False,
            widths=0.2,
        )
        for pc in vparts['bodies']:
            pc.set_facecolor('orchid')
            pc.set_edgecolor('purple')
            pc.set_alpha(0.3)
        if 'cmeans' in vparts:
            vparts['cmeans'].set_color('red')
            vparts['cmeans'].set_linewidth(3)
            vparts['cmeans'].set_linestyle('--')
            vparts['cmeans'].set_alpha(1.0)
            for i, line in enumerate(vparts['cmeans'].get_segments()):
                x_center = i
                y_mean = line[0][1]
                ax_violin.hlines(
                    y=y_mean,
                    xmin=x_center-0.4,
                    xmax=x_center+0.4,
                    colors='red',
                    linewidths=2,
                    linestyle='-',
                    alpha=0.7
                )

        # ax_violin.set_xlim(-0.5, C-0.5)
        ax_violin.set_title('Channel-wise Distribution')
        ax_violin.set_xlabel('Channels')
        ax_violin.set_ylabel('Values')


        # 4. Fixed Hexbin Plot
        ax_hex = fig.add_subplot(gs[1:, 2])
        
        # Fix bins parameter
        hex_bins = 'linear' if has_negative else 'log'
        if has_negative or (data_max - data_min < 1e-6):
            hex_bins = 50  # Fallback to linear bins
            
        hb = ax_hex.hexbin(
            x=flat_values,
            y=channel_coords,
            gridsize=(50, C//2),
            cmap='coolwarm' if has_negative else 'plasma',
            bins=hex_bins,
            mincnt=1,
            extent=(data_min, data_max, 0, C)
        )
        plt.colorbar(hb, ax=ax_hex, label='Density')
        ax_hex.set_title('Value-Channel Density')
        ax_hex.set_xlabel('Values')
        ax_hex.set_ylabel('Channels')

        plt.tight_layout()
        plt.show()

def simple_value_distribution(event_matrix, title="Value Distribution Analysis", font_scale=1.0):
    """
    Visualize event value distribution across multiple dimensions for normalization
    Simplize of the above one.
    """
    if isinstance(event_matrix, torch.Tensor):
        event_matrix = event_matrix.numpy()


    original_rc = {
        'font.size': get_rc_float('font.size'),
        'axes.titlesize': get_rc_float('axes.titlesize'),
        'axes.labelsize': get_rc_float('axes.labelsize'),
        'xtick.labelsize': get_rc_float('xtick.labelsize'),
        'ytick.labelsize': get_rc_float('ytick.labelsize'),
        'figure.titlesize': get_rc_float('figure.titlesize')
    }
    
    with plt.rc_context({
        'font.size': original_rc['font.size'] * font_scale,
        'axes.titlesize': original_rc['axes.titlesize'] * font_scale,
        'axes.labelsize': original_rc['axes.labelsize'] * font_scale,
        'xtick.labelsize': original_rc['xtick.labelsize'] * font_scale,
        'ytick.labelsize': original_rc['ytick.labelsize'] * font_scale,
        'figure.titlesize': original_rc['figure.titlesize'] * font_scale
    }):


        T = event_matrix.shape[0]
        if event_matrix.ndim > 2:
            event_matrix = event_matrix.reshape(T, -1)
        
        C = event_matrix.shape[1]
        flat_values = event_matrix.flatten()
        time_coords = np.repeat(np.arange(T), C)
        channel_coords = np.tile(np.arange(C), T)
        

        data_min = np.min(flat_values)
        data_max = np.max(flat_values)
        has_negative = data_min < 0


        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 2, figure=fig, 
                      width_ratios=[1, 2], 
                      height_ratios=[1, 1.5]) 
        fig.suptitle(title, fontsize=14 * font_scale)


        ax_hist = fig.add_subplot(gs[0, 0])
        # if has_negative:
        #     bins = np.linspace(data_min, data_max, 50)
        #     yscale = 'linear'
        # else:
        #     bins = np.geomspace(max(1e-6, data_min), data_max, 50)
        #     # bins = np.logspace(max(1e-6, data_min), data_max, 50)
        #     yscale = 'log'

        bins = np.linspace(data_min, data_max, 50)
        yscale = 'linear'
        
        counts, bins, _ = ax_hist.hist(flat_values, bins=bins, 
                                    color='skyblue', edgecolor='navy')
        

        percentiles = np.percentile(flat_values, [25, 50, 75, 95, 99])
        for p in percentiles:
            ax_hist.axvline(p, color='red', linestyle='--', alpha=0.7)
        
        ax_hist.set_title(f'Value Distribution Histogram ({yscale} Scale)')
        ax_hist.set_xlabel('Values')
        ax_hist.set_ylabel('Count')
        ax_hist.set_yscale(yscale)
        ax_hist.grid(True, alpha=0.3)


        ax_time = fig.add_subplot(gs[0, 1])
        cmap = 'coolwarm' if has_negative else 'viridis'
        sc = ax_time.scatter(time_coords, flat_values, c=channel_coords,
                            cmap=cmap, alpha=0.3, s=5, 
                            vmin=0, vmax=C-1)
        plt.colorbar(sc, ax=ax_time, label='Channel ID')
        ax_time.set_title('Temporal Distribution')
        ax_time.set_xlabel('Time Steps')
        ax_time.set_ylabel('Values')
        ax_time.grid(True, alpha=0.3)

        ax_hex = fig.add_subplot(gs[1, :]) 
        
        gridsize_x = 50
        gridsize_y = min(50, C)
        
        hex_bins = 'log'
        if has_negative or (data_max - data_min < 1e-6):
            hex_bins = 50  
            
        hb = ax_hex.hexbin(
            x=flat_values,
            y=channel_coords,
            gridsize=(gridsize_x, gridsize_y),
            cmap='coolwarm' if has_negative else 'plasma',
            bins=hex_bins,
            mincnt=1,
            extent=(data_min, data_max, 0, C-1)
        )
        plt.colorbar(hb, ax=ax_hex, label='Density')
        ax_hex.set_title('Value-Channel Density Map')
        ax_hex.set_xlabel('Values')
        ax_hex.set_ylabel('Channel ID')
        ax_hex.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

def plot_time_step_distribution(dataset, title="Time Steps Distribution", font_scale=1.0):
    """
    Draw the statistical map of time step of dataset
    Args:
        dataset: input dataset, the shape of each sample is [T, ...]
        title: title of graph
    """
    time_steps = []
    
    for sample in tqdm(dataset, desc="Collecting time steps", leave=False):

        t = sample[0].shape[0]  
        time_steps.append(t)
    
    ts_array = np.array(time_steps)
    
    mean_val = np.mean(ts_array)
    std_val = np.std(ts_array)
    median_val = np.median(ts_array)
    
    plt.figure(figsize=(10, 6))
    
    n, bins, patches = plt.hist(ts_array, bins=50, 
                            color='skyblue', 
                            edgecolor='black',
                            alpha=0.7)
    textstr = '\n'.join((
        f'Mean = {mean_val:.2f}',
        f'Std = {std_val:.2f}',
        f'Median = {median_val:.2f}',
        f'Total Samples = {len(ts_array)}'))
    
    plt.gca().text(0.75, 0.95, textstr, transform=plt.gca().transAxes,
                fontsize=12 * font_scale, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    

    plt.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean ({mean_val:.2f})')
    plt.axvline(median_val, color='g', linestyle=':', linewidth=2, label=f'Median ({median_val:.2f})')
    
    plt.title(title, fontsize=14 * font_scale)
    plt.xlabel('Time Steps (T)', fontsize=12 * font_scale)
    plt.ylabel('Frequency', fontsize=12 * font_scale)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_event_distribution(frames, title="Event Distribution Analysis", font_scale=1.0):
    """
    Visualize event data distribution with multiple complementary plots

    The input is frame after tonic.transform.ToFrame.

    This function should work only ToFrame transform is used. If you use other transforms, 
    the results could not be accuracy.
    
    Args:
        frames (np.ndarray or Tensor): Input data of shape [Time_steps, Channels]
        title (str): Title for the visualization
    """
    import matplotlib.colors as mcolors

    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()

    T, C = frames.shape[0], frames.shape[1]
    if frames.ndim > 2:
        frames = frames.reshape(T, C, -1)

    plot_data = frames.copy().astype(np.float32)
    plot_data_flat = plot_data.reshape(-1)
    np.clip(plot_data_flat, a_min=1e-6, a_max=None, out=plot_data_flat) 

    original_rc = {
        'font.size': get_rc_float('font.size'),
        'axes.titlesize': get_rc_float('axes.titlesize'),
        'axes.labelsize': get_rc_float('axes.labelsize'),
        'xtick.labelsize': get_rc_float('xtick.labelsize'),
        'ytick.labelsize': get_rc_float('ytick.labelsize'),
        'figure.titlesize': get_rc_float('figure.titlesize')
    }
    
    with plt.rc_context({
        'font.size': original_rc['font.size'] * font_scale,
        'axes.titlesize': original_rc['axes.titlesize'] * font_scale,
        'axes.labelsize': original_rc['axes.labelsize'] * font_scale,
        'xtick.labelsize': original_rc['xtick.labelsize'] * font_scale,
        'ytick.labelsize': original_rc['ytick.labelsize'] * font_scale,
        'figure.titlesize': original_rc['figure.titlesize'] * font_scale
    }):

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(title, fontsize=14 * font_scale)
        gs = fig.add_gridspec(3, 3)

        # Main image - 2D event distribution
        ax1 = fig.add_subplot(gs[:2, :2])
        hb = ax1.hexbin(
            x=np.tile(np.arange(T), C),
            y=np.repeat(np.arange(C), T),
            C=plot_data_flat,
            gridsize=50,
            cmap='viridis',
            norm=mcolors.LogNorm(
                vmin=max(1e-5, plot_data_flat.min()),
                vmax=plot_data_flat.max(),
            ),   # Dynamically set the minimum value
            mincnt=1 
        )
        ax1.set_xlabel('Time Steps', fontsize=10 * font_scale)
        ax1.set_ylabel('Channels', fontsize=10 * font_scale)
        cb = fig.colorbar(hb, ax=ax1, pad=0.02)
        cb.set_label('Log-scaled Event Count', rotation=270, labelpad=20)

        # Time dimension distribution (bottom)
        ax2 = fig.add_subplot(gs[2, :2])
        time_sum = frames.reshape(T, -1).sum(axis=1)
        ax2.plot(time_sum, color='tab:blue')
        ax2.fill_between(np.arange(T), time_sum, alpha=0.3)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Total Events')
        ax2.set_title('Temporal Distribution', fontsize=10 * font_scale)
        ax2.grid(True, alpha=0.3)

        # Channel dimension distribution (right)
        ax3 = fig.add_subplot(gs[:2, 2])
        chan_sum = frames.reshape(-1, C).sum(axis=0)
        ax3.plot(chan_sum, np.arange(C), color='tab:orange')
        ax3.fill_betweenx(np.arange(C), chan_sum, alpha=0.3)
        ax3.set_xlabel('Total Events')
        ax3.set_ylabel('Channels')
        ax3.set_title('Channel Distribution', fontsize=10 * font_scale)
        ax3.grid(True, alpha=0.3)

        # Statistics table (bottom right)
        ax4 = fig.add_subplot(gs[2, 2])
        stats = {
            'Total Events': f"{frames.sum():,}",
            'Mean/Time Step': f"{frames.mean(axis=1).mean():.1f} ± {frames.mean(axis=1).std():.1f}",
            'Density (%)': f"{100 * np.mean(frames > 0):.2f}%",
            'Max Channel': f"Ch{np.argmax(chan_sum)} ({chan_sum.max():,})",
            'Peak Time': f"T{np.argmax(time_sum)} ({time_sum.max():,})"
        }
        ax4.table(
            cellText=[[v] for v in stats.values()],
            rowLabels=list(stats.keys()),
            loc='center',
            cellLoc='left',
            bbox=[0.1, 0.2, 0.8, 0.6],
            colWidths=[0.6]
        )
        ax4.axis('off')

        plt.tight_layout()
        plt.show()

def get_rc_float(param_name):
    val = plt.rcParams[param_name]
    if isinstance(val, (tuple, list)):
        return float(val[0])
    if isinstance(val, str):
        from matplotlib import font_manager
        return font_manager.FontProperties(size=val).get_size_in_points()
    return float(val)