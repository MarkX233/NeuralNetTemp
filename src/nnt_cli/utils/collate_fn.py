from tonic.collation import PadTensors
import torch



class FixedLengthPadTensors(PadTensors):
    """
    Custom collate function, used to unify the time length of samples to a fixed length (fixed_length).
    Samples that exceed the length are truncated and insufficient samples are padded with 0.
    
    Args:
        fixed_length (int): Fixed length of time.
        batch_first (bool): Whether to use batch_size as the first dimension.
    """

    def __init__(self, fixed_length: int, batch_first: bool = True):
        super().__init__(batch_first=batch_first)
        self.fixed_length = fixed_length

    def __call__(self, batch):
        samples_output = []
        labels_output = []

        for sample, label in batch:
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)

            if sample.shape[0] > self.fixed_length:
                sample = sample[:self.fixed_length]
            else:
                padding = torch.zeros(
                    (self.fixed_length - sample.shape[0], *sample.shape[1:]),
                    device=sample.device
                )
                sample = torch.cat((sample, padding), dim=0)

            samples_output.append(sample)
            labels_output.append(label)

        samples_output = torch.stack(samples_output, 0 if self.batch_first else 1)
        if len(labels_output[0].shape) > 1:
            labels_output = torch.stack(labels_output, 0 if self.batch_first else -1)
        else:
            labels_output = torch.tensor(labels_output, device=label.device)

        return (samples_output, labels_output)
    

class PadTensorsWithMinus1():
    """
    This is a custom collate function for a pytorch dataloader to load multiple event recordings
    at once. It's intended to be used in combination with sparse tensors. All tensor sizes are
    extended to the largest one in the batch, i.e. the longest recording.

    Example:
        >>> dataloader = torch.utils.data.DataLoader(dataset,
        >>>                                          batch_size=10,
        >>>                                          collate_fn=tonic.collation.PadTensors(),
        >>>                                          shuffle=True)

    Above is the original docstring of tonic.collation.PadTensor
    I modified it to pad value -1 when bit_width is 1, when bit_width is other, pad 0.
    So the data can fit brevitas binary quant.
        
    """

    def __init__(self, batch_first: bool = True, bit_width=1):
        self.batch_first = batch_first
        self.bit_width=bit_width

    def __call__(self, batch):
        samples_output = []
        targets_output = []

        max_length = max([sample.shape[0] for sample, target in batch])
        for sample, target in batch:
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            if sample.is_sparse:
                sample.sparse_resize_(
                    (max_length, *sample.shape[1:]),
                    sample.sparse_dim(),
                    sample.dense_dim(),
                )
            else:
                sample = torch.cat(
                    (
                        sample,
                        torch.zeros(
                            max_length - sample.shape[0],
                            *sample.shape[1:],
                            device=sample.device
                        ) if self.bit_width>1 else
                        torch.full(
                            (max_length - sample.shape[0], *sample.shape[1:]),
                            -1,
                            device=sample.device
                        ),
                    )
                )
            samples_output.append(sample)
            targets_output.append(target)
        
        samples_output = torch.stack(samples_output, 0 if self.batch_first else 1)
        if len(targets_output[0].shape) > 1:
            targets_output = torch.stack(targets_output, 0 if self.batch_first else -1) 
        else:
            targets_output = torch.tensor(targets_output, device=target.device)
        return (samples_output, targets_output)
    
class TensorTransposeBatch():
    def __init__(self, batch_first: bool = False):
        self.batch_first = batch_first

    def __call__(self, batch):
        if self.batch_first is True:
            return batch
        
        else:
            samples_output = []
            targets_output = []
            for sample, target in batch:

                if not isinstance(sample, torch.Tensor):
                    sample = torch.tensor(sample)
                if not isinstance(target, torch.Tensor):
                    target = torch.tensor(target)

                sample.transpose(1,0)

                samples_output.append(sample)
                targets_output.append(target)
            
            samples_output = torch.stack(samples_output, 0 if self.batch_first else 1)
            if len(targets_output[0].shape) > 1:
                targets_output = torch.stack(targets_output, 0 if self.batch_first else -1) 
            else:
                targets_output = torch.tensor(targets_output, device=target.device)
            return (samples_output, targets_output)