import numpy as np
import torch

class FlattenTransform():
    def __init__(self):
        pass
    def __call__(self, frame):
        # Input: for nminst [time_length, 2*34*34]
        frame = frame.reshape(frame.shape[0], -1)
        return frame
    
class ToFloat():
    def __init__(self):
        pass
    def __call__(self, frame):
        frame = frame.float()
        return frame
    
class AddChannel():
    def __init__(self):
        # Add Channel for Normalize..
        pass
    def __call__(self, frame):
        frame = frame.view(frame.shape[0], 1 , -1)
        return frame
    
class PrintTransform():
    def __init__(self):
        pass
    def __call__(self, sample):
        print(type(sample))
        print(sample.shape)
        print(sample[0].shape)
        return sample
    
class DynamicNormalize():
    def __init__(self):
        """
        Only normalize one frame in its own scale.
        If the scale between samples is inconsistent, models may not be able to distinguish the absolute difference in event density.
        """
        pass
    def __call__(self, tensor):
        
        tensor = tensor / tensor.max()
        return tensor

class ChannelPercentileNorm():
    """
    Normalize by the percentile value of each channel of frame in time scale.

    This approach is often used in standardized input data in the pulse neural network, 
    especially suitable for sparse but strong peak values in pulse data.
    """
    def __init__(self, percentile=95, abs=False):
        self.percentile = percentile
        self.abs=abs
    def __call__(self, x):
        # Calculate the radical benchmark value of each channel
        # x: [T, C]
        if self.abs is True:
            convert_x=np.abs(x)
        else:
            convert_x=x
        ref = np.percentile(convert_x, self.percentile, axis=0, keepdims=True)
        return x / np.maximum(ref, 1e-6)

class RandomChannelDropout():
    """
    Randomly mask out entire frequency channels to improve robustness.
    
    Args:
        dropout_prob (float): Probability of dropping a channel (0.0-1.0)
        mask_value (float): Value to set for dropped channels
    """
    def __init__(self, dropout_prob=0.2, mask_value=0.0):
        self.dropout_prob = dropout_prob
        self.mask_value = mask_value
        
    def __call__(self, frames):
        """
        Args:
            frames (np.ndarray): Input event frames of shape [T, C]
            
        Returns:
            np.ndarray: Masked frames of same shape [T, C]
        """
        # Generate channel mask
        channel_mask = np.random.rand(frames.shape[1]) > self.dropout_prob
        
        # Apply mask while preserving temporal dimension
        masked_frames = frames.copy()
        masked_frames[:, ~channel_mask] = self.mask_value
        
        return masked_frames

class QuantlieClip():
    def __init__(self, quantlie=0.98):
        self.quantlie = quantlie
    def __call__(self, frames):
        threshold = np.quantile(frames, self.quantlie)
        clipped_frames = np.clip(frames, a_min=None, a_max=threshold)
        return clipped_frames
    
class ZeroNegTransform():
    """
    In order to let brevitas 1-bit quantization work, the value 0 must be transformed to a negative value.
    This is because the quantization function in brevitas will quantize the value 0 to 1.
    """
    def __init__(self, negative_value=-1, onebit_en=False):
        self.negative_value = negative_value
        self.onebit_en = onebit_en
        
    def __call__(self, frames):
        if self.onebit_en:
            frames_copy = frames.copy() if isinstance(frames, np.ndarray) else frames.clone()
            neg_frames = (frames_copy <= 0) * self.negative_value + (frames_copy > 0) * frames_copy
            return neg_frames
        else:
            return frames

class BinnedTransform:
    def __init__(self, n_bins, time_first=True):
        """
        Transform for Binned Time Series Data.
        binned_len = C // self.n_bins
        
        This transform comes form the paper:
        I. Hammouamri, I. Khalfaoui-Hassani, and T. Masquelier, "Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings," in Proc. Twelfth Int. Conf. Learn. Representations (ICLR), 2024. [Online]. Available: https://openreview.net/forum?id=4r2ybzJnmN

        Args:
        n_bins (int): Number of bins to divide the input data into.
        time_first (bool): If True, the first dimension is time. If False, the first dimension is channel.
        """
        self.n_bins = n_bins
        self.time_first = time_first

    def __call__(self, data):

        if isinstance(data, np.ndarray):
            return self._process_numpy(data)
        elif isinstance(data, torch.Tensor):
            # Convert to numpy for processing
            data = data.numpy()
            return torch.tensor(self._process_numpy(data))
        else:
            raise TypeError("Input must be the type of numpy.ndarray or torch.Tensor")

    def _process_numpy(self, x: np.ndarray) -> np.ndarray:
        if self.time_first:
            T, C = x.shape
        else:
            C, T = x.shape
        binned_len = C // self.n_bins
        binned_frames = np.zeros((x.shape[0], binned_len))
        for i in range(binned_len):
            if self.time_first:
                binned_frames[:,i] = x[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)
            else:
                binned_frames[:,i] = x[i*self.n_bins : (i+1)*self.n_bins, :].sum(axis=0)

        return binned_frames
