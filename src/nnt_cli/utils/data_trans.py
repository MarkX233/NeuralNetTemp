from brevitas.quant_tensor.base_quant_tensor import QuantTensor
import torch

def float_quant_tensor2int(quant_tensor,clone=True, int_datatype=None, round=False):
    """
    Transform float quant tensor to int number.
    According to the brevitas source code, quant tensor is a Tuple, whose form is:
    (value, scale, zero_point, ...)
    The int tensor is calculated by:
    int_value = value / scale   # Using broadcasting
    int_value = int_value + zero_point

    Args:
        quant_tensor: The input quant tensor, which is a tuple of (value, scale, zero_point, ...)
        clone: Whether to clone the tensor or not. Default is True.
        int_datatype: The datatype of the output int tensor. Default is None. \\
                1. If None, the output tensor will be the same as the input tensor. \\
                2. If str, the output tensor will be converted to the specified type. \\
                3. If torch.dtype, the output tensor will be converted to the specified type.
        round: Whether to round the int value or not. Default is False.

    Return: Tensor
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
        if round:
            int_value = int_value.round()

        if int_datatype is not None:
            if isinstance(int_datatype, str):
                match int_datatype:
                    case "int":
                        int_value = int_value.to(torch.int)
                    case "int8":
                        int_value = int_value.to(torch.int8)
                    case "uint8":
                        int_value = int_value.to(torch.uint8)
                    case "int16":
                        int_value = int_value.to(torch.int16)
                    case "uint16":
                        int_value = int_value.to(torch.uint16)
                    case "int32":
                        int_value = int_value.to(torch.int32)
                    case "uint32":
                        int_value = int_value.to(torch.uint32)
                    case "int64":
                        int_value = int_value.to(torch.int64)
                    case "uint64":
                        int_value = int_value.to(torch.uint64)
                    case _:
                        raise ValueError("Wrong input! The input string is not a valid." \
                        "Try use directly torch.int8, torch.uint8, etc.")
            elif isinstance(int_datatype, torch.dtype):
                int_value = int_value.to(int_datatype)
            else:
                raise ValueError("Wrong input type! The input is not a valid type." \
                "Try use string or torch.dtype.")
            
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