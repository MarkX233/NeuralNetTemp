import librosa
import numpy as np
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
    
