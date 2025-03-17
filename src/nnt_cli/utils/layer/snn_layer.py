import torch

from nnt_cli.utils.data_trans import *

class RateEncodingLayer(torch.nn.Module):
    def __init__(self, num_steps=10):
        super().__init__()
        self.num_steps=num_steps
    def forward(self, images): # self-made layer, forward arrguments only contain x
        batch_size, _, height, width = images.shape
        images = images.view(batch_size, -1)  # [batch_size, 784]
        spike_sequence = torch.bernoulli(images.unsqueeze(0).repeat(self.num_steps, 1, 1))  # [time_steps, batch_size, 784]
        return spike_sequence.view(self.num_steps, batch_size, height*width)
    
class QuantTensorToTensorLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        out=float_quant_tensor2int(input,clone=False)
        return out
    
class QuantTensorToTensorLayer2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        out=float_quant_tensor2int2(input,clone=False,float_datatype=True)
        return out

class QuantTensorToTensorLayer3(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        out=float_quant_tensor2int3(input,clone=False,float_datatype=True)
        return out