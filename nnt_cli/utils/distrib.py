import torch


def multi_GPU_distrib(net, debug_mode=True):
    """
    Optimization for multi-GPU in HPC
    When using multiple GPUs, you can increase the batch size appropriately to improve GPU utilization.
    """
    if debug_mode is True:
        # Do nothing
        print("Using Debug mode for GPU distributor.")
        return net
    else:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print("Using DataParallel for multi-GPU training.")
                net = torch.nn.DataParallel(net)
                return net