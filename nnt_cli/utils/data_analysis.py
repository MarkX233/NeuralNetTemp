from tqdm import tqdm
import numpy as np
import torch


def compute_mean_std_loader(loader, no_channel=False):
    """
    Calculate the global average and standard deviation per channel of all time steps, all batches

    Input format requirements:
        -In the first three -dimensional fixation is [Time Step T, Batch B, Channel C]
        -The subsequent dimension is any space dimension (such as h, w or d, h, w, etc.)
        For example: NMNIST: frames.shape: [T, Batch, C, H, W]
    Return:
        -mean: The average value of each channel (shape [C])
        -std: The standard deviation value of each channel (shape [C])

    """
    sum_pixels = None
    sum_squared_pixels = None
    total_count = 0

    for frames, _ in tqdm(loader,unit="Batch"):

        if frames.dim() < 3:
            raise ValueError(f"Input must be at least 3D [T, B, C, ...], got shape {frames.shape}")
        
        T, B = frames.shape[0], frames.shape[1]

        if no_channel is True:
            C = 1
            frames_flat = frames.view(T * B, -1)
        else: 
            C = frames.shape[2]
            frames_flat = frames.view(T * B, C, -1)

        if sum_pixels is None:
            device = frames.device
            sum_pixels = torch.zeros(C, device=device)
            sum_squared_pixels = torch.zeros(C, device=device)

        sum_pixels += frames_flat.sum(dim=[0, -1])          # [C]
        sum_squared_pixels += (frames_flat ** 2).sum(dim=[0, -1])  # [C]
        total_count += frames_flat.shape[0] * frames_flat.shape[-1]  # (T*B) * spatial_size

    mean = sum_pixels / total_count
    std = torch.sqrt((sum_squared_pixels / total_count) - mean.pow(2))

    return mean, std

def compute_mean_std_dataset(dataset):
    all_frames = []
    for frame, label in tqdm(dataset):
        all_frames.append(frame)


    all_frames = np.concatenate(all_frames, axis=0)  # Merge to [Total_t, C, H, W]
    

    channel_means = []
    channel_stds = []
    for c in range(all_frames.shape[1]):
        channel_data = all_frames[:, c, :, :]
        channel_means.append(np.mean(channel_data))
        channel_stds.append(np.std(channel_data))
        
    return channel_means, channel_stds


def compute_min_max_loader(loader, no_channel=False):
    """
    Calculate the global minimum value and maximum value of each channel
    Input format requirements:
        -In the first three -dimensional fixation is [Time Step T, Batch B, Channel C]
        -The subsequent dimension is any space dimension (such as h, w or d, h, w, etc.)
    Return:
        -min_values: The minimum value of each channel (shape [C])
        -max_values: The maximum value of each channel (shape [C])
    """
    min_values = None
    max_values = None

    for frames, _ in tqdm(loader, unit="Batch"):
        if frames.dim() < 3:
            raise ValueError(f"Input must be at least 3D [T, B, C, ...], got shape {frames.shape}")

        T, B = frames.shape[0], frames.shape[1]

        if no_channel is True:
            C = 1
            frames_flat = frames.view(T * B, -1)
        else: 
            C = frames.shape[2]
            frames_flat = frames.view(T * B, C, -1)

        if min_values is None:
            device = frames.device
            min_values = torch.full((C,), float('inf'), device=device)
            max_values = torch.full((C,), -float('inf'), device=device)

        batch_min = frames_flat.min(dim=-1).values.min(dim=0).values  # [C]
        batch_max = frames_flat.max(dim=-1).values.max(dim=0).values  # [C]

        min_values = torch.minimum(min_values, batch_min)
        max_values = torch.maximum(max_values, batch_max)

    return min_values, max_values

def analyze_time_length(dataset):
    """
    Analyze the time frame length distribution of the dataset

    Args:
        dataSet (dataset): Pre -processing data set object, each sample format is (events, label)
    
    Return:
        Dictionary containing statistical indicators:
        {
            'min': minimum number of frames,
            'max': maximum time frame number,
            'mean': Average time frame number,
            'std': Standard deviation,
            'total_samples': Total sample number
        }
    """

    lengths = [sample[0].shape[0] for sample in dataset]

    return {
        'min': np.min(lengths),
        'max': np.max(lengths),
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'total_samples': len(lengths)
    }

def check_gradient_norm(self):
        grad_norms = [p.grad.norm().item() for p in self.net.parameters()]
        print(f"Gradient norm: {np.mean(grad_norms):.3e} ± {np.std(grad_norms):.3e}, the normal value should be between 1E-6 and 1E-3")
