import librosa
import numpy as np
import torch
import snntorch as snn

"""
This file contains transforms in time domain
"""

class FixedTimeStepTrans():
    """
    Fix time steps of frames, using central interception + repeat filling.
    Used in transforms.

    The input type should be numpy.array.

    Different from FixedLengthPadTensors below. That one is collate_fn, used in dataloader.
    And Principle is also slightly different.

    """
    def __init__(self, target_frames=100):
        self.target = target_frames
        
    def __call__(self, frames):
        T = frames.shape[0]
        # T for time_steps, C for contents.
        
        if T == self.target:
            return frames
        elif T > self.target:
            # Central interception strategy
            start = (T - self.target) // 2
            return frames[start:start+self.target]
        else:
            # Repeat filling
            repeats = (self.target + T - 1) // T
            tile_dims = (repeats,) + (1,)*(frames.ndim-1)
            padded = np.tile(frames, tile_dims)
            return padded[:self.target]

class MelSpectrogramEncoder():
    """
        In traditional voice processing, Mel's spectrum can convert the linear frequency 
        into the scales of Mel by simulating human ear hearing special characteristics 
        to better capture the relevant characteristics of voice.
    """
    def __init__(self, n_mels=64):
        self.n_mels = n_mels
    
    def __call__(self, frames):

        frames_transposed = frames.T

        nfft=(frames_transposed.shape[0]-1)*2

        mel_filters = librosa.filters.mel(
            sr=16000, 
            n_fft=nfft,
            n_mels=self.n_mels,
            fmax=8000,
            fmin=50
        )

        mel_energy = np.dot(mel_filters, frames_transposed)
            
        return np.log1p(mel_energy.T)
    
class DeltaCalculator():
    """
    By calculating the differences between adjacent time frames, add another dimension of 
    the dynamic changes of characteristics of the voice.

    Args:
        order (int): The number of dimension be added.
        window (int): The size of delta window.

    Return:
        Extended features.

        For example:
            The input is the frames after mel filters.
            Input features shape: [time_steps, n_features]
            Output: [time_steps, n_features + order * delta_features], and len(delta_features)==len(n_features)
    """
    def __init__(self, order=2, window=2):
        self.order = order
        self.window = window

    def __call__(self, features):
        features = features.astype(np.float64)
        deltas = [features]
        for _ in range(self.order):
            delta = np.zeros_like(deltas[-1])
            for t in range(self.window, features.shape[0]-self.window): # (window, time_steps-window)
                numerator = sum(n*(deltas[-1][t+n] - deltas[-1][t-n]) 
                               for n in range(1, self.window+1))
                denominator = 2 * sum(n**2 for n in range(1, self.window+1))
                delta[t] = numerator / denominator
            deltas.append(delta)
        return np.concatenate(deltas, axis=-1) 

class RandomTimeWarp():
    """
    Non -linear distortion of time axis, changes in the speed of simulation language
    Args:
        max_shift: The maximum time frame displacement
        num_control_points (int): Number of control points for the warp curve
    """
    def __init__(self, max_shift=3, num_control_points=5):
        self.max_shift = max_shift
        self.num_control_points = num_control_points
        
    def __call__(self, frames):
        """
        Args:
            frames (np.ndarray): Input event frames of shape [T, C]
            
        Returns:
            np.ndarray: Warped frames of same shape [T, C]
        """
        T, C = frames.shape
        
        # Generate control points with random shifts
        control_x = np.linspace(0, T-1, self.num_control_points).astype(int)
        control_y = control_x + np.random.randint(-self.max_shift, self.max_shift+1, 
                                                 size=self.num_control_points)
        
        # Generate full warp curve using linear interpolation
        warp_curve = np.interp(np.arange(T), control_x, np.clip(control_y, 0, T-1))
        warp_curve = warp_curve.astype(int)
        
        return frames[warp_curve]
    
class ProbabilisticBinarize():
    def __init__(self, min_prob=0.1, max_prob=0.9, scale_type="linear", gamma=1.0, exclude_min=True):
        """
        Probabilistic binarization of input data based on a probability map.
        The probability map is generated based on the input data's min and max values,
        and the specified scaling type.

        linearly mapped data x:
        x = (data - data_min) / data_range

        linear	    p = min_p + (max_p - min_p) * x	\\
        exp	        p = min_p + (max_p - min_p) * (e^(γx)-1)/(e^γ-1) \\
        log	        p = min_p + (max_p - min_p) * logγ(1+x(γ-1)) \\
        root	    p = min_p + (max_p - min_p) * x^(1/γ)	\\
        power	    p = min_p + (max_p - min_p) * x^γ	

        Args:
            min_prob (float): Minimum trigger probability (0.0~1.0)
            max_prob (float): Maximum trigger probability (0.0~1.0)
            scale_type (str): Probability map curve type ["linear", "exp", "log", "sqrt", "quadratic"]
            gamma (float): Nonlinear coefficient for the probability map curve
            exclude_min (bool): Whether to exclude the minimum value from the probability map
        """
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.scale_type = scale_type
        self.gamma = gamma
        self.exclude_min = exclude_min
        
        assert scale_type in ["linear", "exp", "log", "sqrt", "quadratic"]
        assert 0 <= min_prob <= max_prob <= 1.0

        if scale_type == "log":
            assert gamma > 1.0, "gamma must >1 for log scaling"
        elif scale_type == "root":
            assert gamma > 1.0, "gamma must >1 for root scaling"

    def __call__(self, data_matrix):
        """
        Args:
            data_matrix (np.ndarray or torch.Tensor): Input data matrix of shape [T, C]
        """
        if isinstance(data_matrix, torch.Tensor):
            data_np = data_matrix.numpy()
        else:
            data_np = data_matrix.copy()
            
        original_shape = data_np.shape
        if data_np.ndim > 2:
            data_np = data_np.reshape(original_shape[0], -1)
        
        data_min = data_np.min()
        data_max = data_np.max()
        data_range = data_max - data_min
        if data_range == 0:
            return torch.zeros_like(data_matrix) if isinstance(data_matrix, torch.Tensor) \
                   else np.zeros_like(data_matrix)

        # Data linearly mapped to [0,1] interval    
        normalized = (data_np - data_min) / data_range
        
        if self.scale_type == "linear":
            scaled = normalized
        elif self.scale_type == "exp":
            scaled = np.exp(self.gamma * normalized) - 1
            scaled = scaled / (np.exp(self.gamma) - 1)
        elif self.scale_type == "log":
            scaled = np.log1p(normalized * (self.gamma - 1))
            scaled = scaled / np.log(self.gamma)
        elif self.scale_type == "root":
            scaled = normalized ** (1.0 / self.gamma)
        elif self.scale_type == "power":
            scaled = normalized ** self.gamma

        # Generate boolean mask to exclude minimum value
        if self.exclude_min:
            non_min_mask = (scaled > 0).astype(np.float32)
        else:
            non_min_mask = np.ones_like(scaled, dtype=np.float32)

        prob_map = self.min_prob*non_min_mask + (self.max_prob - self.min_prob) * scaled
        
        prob_map = np.clip(prob_map, 0.0, 1.0) # In case of numerical instability
        binary_np = np.random.binomial(n=1, p=prob_map).astype(np.int8)
        
        # Restore original shapes
        binary_np = binary_np.reshape(original_shape)
        if isinstance(data_matrix, torch.Tensor):
            return torch.from_numpy(binary_np)
        else:
            return binary_np

class LIFTransform:
    def __init__(
        self,
        threshold=1.0,      
        beta=0.9,
        reset_mechanism="zero",
        normalize_input=True,    # Whether to automatically normalize input to [0,1]
    ):
        self.normalize_input = normalize_input
        # self.mem = None
        self.beta = beta
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        self.lif = None

    def _init_lif(self):
        self.lif = snn.Leaky(
            beta=self.beta,
            threshold=self.threshold,
            reset_mechanism=self.reset_mechanism,
            init_hidden=True
        )

    def __call__(self, frames):

        if self.lif is None:
            self._init_lif()

        self.lif.reset_mem()

        if not isinstance(frames, torch.Tensor):
            frames_tensor = torch.as_tensor(frames, dtype=torch.float32)
        else:
            frames_tensor = frames.clone().detach()
        
        frames_shape = frames_tensor.shape

        if frames_tensor.dim() == 1:
            frames_tensor = frames_tensor.unsqueeze(-1)  # [T] → [T, 1]
            print("Warning: Input tensor has only 1 dimension. "
            "The time dimension is assumed to be the first dimension."
            "Which means the input tensor does not have a channel dimension. "
            "Please check the input tensor shape.")

        if self.normalize_input:
            frames_tensor = (frames_tensor - frames_tensor.min()) / (
                frames_tensor.max() - frames_tensor.min() + 1e-8
            )

        spike_frame = []

        for t in range(frames_tensor.size(0)):
            spk = self.lif(frames_tensor[t])
            spike_frame.append(spk)

        spike_frame = torch.stack(spike_frame).view(frames_shape)

        return spike_frame
