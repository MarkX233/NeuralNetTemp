import torch

from brevitas.core.quant.float import FloatQuant
from brevitas.core.quant.binary import BinaryQuant
from brevitas.core.scaling import *
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import QuantType
from brevitas.core.function_wrapper import Identity
from brevitas.core.function_wrapper.clamp import FloatClamp, TensorClamp

from brevitas.quant.binary import SignedBinaryWeightPerTensorConst, SignedBinaryActPerTensorConst

from brevitas.proxy import ActQuantProxyFromInjector


class FloatQuantizerBeta(ExtendedInjector):
    """
    Quantizer for beta parameter in Leaky layer.

    Usage::
        FloatQuantizerBeta.tensor_quant(tensor)
        Returns: A Tuple with quantized tensor.
    
    Warning: This class can not be instantiated directly and the args can't be changed outside. 
             Using @classmethod or @staticmethod still can't change the args.
             The only way to change the args is to change the class attributes directly or creating another class and inherit.
    """
    quant_type = QuantType.FP

    tensor_quant = FloatQuant
    scaling_impl = ConstScaling
    scaling_init = 0.5
    scaling_min_val = 0.0
    bit_width = 2
    signed = False
    exponent_bit_width = 1
    mantissa_bit_width = 1
    exponent_bias = 0
    input_view_impl = Identity
    float_clamp_impl = FloatClamp
    tensor_clamp_impl = TensorClamp
    max_val = 1.0
    # return_quant_tensor=False
    device = 'cuda:0'
    flag_tensor=torch.tensor(0).to(device)
    return_quant_tensor=True

class FloatQuantizerBeta_NQT(FloatQuantizerBeta):
    """
    Returns not QuantTensor.
    """
    return_quant_tensor=False

class FloatQuantizerThreshold(FloatQuantizerBeta):
    max_val = None
    scaling_impl = ParameterScaling
    scaling_init = 0.5
    bit_width = 2
    exponent_bit_width = 1
    mantissa_bit_width = 1
    exponent_bias = 0



# class FloatWeightQuantizer(FloatQuantizer):
#     proxy_class = WeightQuantProxyFromInjector



from brevitas.inject.defaults import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
# Bit settings, theoretically, the AutoSolver of brevitas should auto change the setting if the receiver is act or weight or etc.
class Act8bit(Int8ActPerTensorFloat):
    quant_type=QuantType.INT
    bit_width=8
    scaling_impl_type='const'
    min_val=-128.0
    max_val=127.0

class Act4bit(Int8ActPerTensorFloat):
    quant_type=QuantType.INT
    bit_width=4
    scaling_impl_type='const'
    min_val=-8.0
    max_val=7.0

class Act2bit(Int8ActPerTensorFloat):
    quant_type=QuantType.INT
    bit_width=2
    scaling_impl_type='const'
    min_val=-2.0
    max_val=1.0

class Act1bit(SignedBinaryActPerTensorConst):
    quant_type=QuantType.BINARY
    bit_width=1
    # scaling_impl_type='const'
    min_val=-1.0
    max_val=1.0

class Weight1bit(SignedBinaryWeightPerTensorConst):
    
    # weight_quant_type=QuantType.BINARY
    # weight_bit_width=1
    # min_val=-1.0
    # max_val=1.0
    pass

class Weight2bit(Int8WeightPerTensorFloat):
    weight_quant_type=QuantType.INT
    weight_bit_width=2

class Weight4bit(Int8WeightPerTensorFloat):
    weight_quant_type=QuantType.INT
    weight_bit_width=4

class Weight8bit(Int8WeightPerTensorFloat):
    weight_quant_type=QuantType.INT
    weight_bit_width=8




class Act1bit_t2(ExtendedInjector):
# From tutorial
    tensor_quant = BinaryQuant
    # tensor_quant = IntQuant
    bit_width=1
    scaling_impl=ConstScaling   	# Dynamic
    scaling_init=0.0
    scaling_min_val = -1.0
    proxy_class = ActQuantProxyFromInjector
    signed = True
    # quant_type=QuantType.BINARY
    

def get_act_quant(bit_width):
    match bit_width:
        case 8:
            return Act8bit
        case 4:
            return Act4bit
        case 2:
            return Act2bit
        case 1:
            return Act1bit
        
def get_weight_quant(bit_width):
    match bit_width:
        case 8:
            return Weight8bit
        case 4:
            return Weight4bit
        case 2:
            return Weight2bit
        case 1:
            return Weight1bit

        
       