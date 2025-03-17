from brevitas.quant_tensor.base_quant_tensor import QuantTensor
import torch

def float_quant_tensor2int(quant_tensor,clone=True):
    """
    Transform float quant tensor to int number.
    According to the brevitas source code, quant tensor is a Tuple, whose form is:
    (value, scale, zero_point, ...)
    Return Tensor
    """
    if isinstance(quant_tensor,(QuantTensor)):
        if clone:
            value=quant_tensor[0].clone()
            scale=quant_tensor[1].clone()
            zero_point=quant_tensor[2].clone()
        else:
            value=quant_tensor[0]
            scale=quant_tensor[1]
            zero_point=quant_tensor[2]

        int_value = value / scale   # Using broadcasting
        int_value = int_value + zero_point
        
        return int_value
    else:
        raise ValueError("Wrong input type! Input is not brevitas quant tensor.")
    
from brevitas.quant_tensor.int_quant_tensor import IntQuantTensor
def float_quant_tensor2int2(quant_tensor,clone=True,float_datatype=False):
    """
    Transform float quant tensor to int number.
    According to the brevitas source code, quant tensor is a Tuple, whose form is:
    (value, scale, zero_point, ...)
    Return Tensor
    """
    if isinstance(quant_tensor,(QuantTensor)):
        if clone:
            qt_i=quant_tensor.clone()
        else:
            qt_i=quant_tensor

        int_value=IntQuantTensor(qt_i[0],qt_i[1],qt_i[2],qt_i[3],qt_i[4],qt_i[5]).int(float_datatype)
        
        return int_value
    else:
        raise ValueError("Wrong input type! Input is not brevitas quant tensor.")


from brevitas.function.ops_ste import round_ste
def float_quant_tensor2int3(quant_tensor,clone=True,float_datatype=False):
    """
    Transform float quant tensor to int number.
    According to the brevitas source code, quant tensor is a Tuple, whose form is:
    (value, scale, zero_point, ...)
    Return Tensor
    """
    if isinstance(quant_tensor,(QuantTensor)):
        if clone:
            qt_i=quant_tensor.clone()
        else:
            qt_i=quant_tensor
        
        int_value = round_ste(pre_round_int_value(qt_i[0],qt_i[1],qt_i[2]))
        if float_datatype:
            # Values at 8bit and lower can be represented exactly with float16 and bfloat16
            # otherwise (e.g. Int16 bias), we upscale to float32
            if qt_i[3] <= 8.:
                return int_value.type(qt_i[1].dtype)
            else:
                return int_value.type(torch.float32)
        else:
            if qt_i[3] <= 8. and qt_i[4]():
                return int_value.to(torch.int8)
            elif qt_i[3] <= 8. and not qt_i[4]():
                return int_value.to(torch.uint8)
            else:
                return int_value.to(torch.int32)
    else:
        raise ValueError("Wrong input type! Input is not brevitas quant tensor.")

def pre_round_int_value(value,scale,zero_point):

    if scale.dtype == torch.bfloat16:
        value = value.type(torch.float32)
        scale = scale.type(torch.float32)
        zero_point = zero_point.type(torch.float32)
    int_value = value / scale
    int_value = int_value + zero_point
    return int_value