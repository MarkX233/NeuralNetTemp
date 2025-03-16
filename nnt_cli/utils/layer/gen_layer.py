import torch
from brevitas.quant_tensor.base_quant_tensor import QuantTensor
import pandas as pd
import os
from torch import nn

from nnt_cli.utils.data_trans import float_quant_tensor2int

class DebugLayer(nn.Module):
    def __init__(self, debug_mode=True):
        super().__init__()
        self.debug_mode = debug_mode

    def forward(self, x):  # Debug layer do nothing but print input
        if self.debug_mode is True:
            print("Debug layer input: ", x)
            print("Debug layer input's shape: ", x.shape)
        return x
    
class InputSaviorLayer(nn.Module):
    def __init__(self, dirpath, filename=None, overwrite=True, sort_by="time",int_quant_tensor=True,squeeze=True):
        """
        Save the input and for network, it does nothing.
        Support input: list or torch.Tensor or QuantTensor from brevitas.
        Args:
            dirpath (str): The directory path where the input data will be saved.
            filename (str): The name of file that the input data will be saved in.
            en (bool): Enable or disable saving the input data.
            overwrite (bool): Whether to overwrite existing data in the directory.
            sort_by (str): The method to sort the saved data, either "time" or "elem". 
                                "time" : Lines are sorted by time. Columns are samples in a batch. 
                                "elem" : Lines are sorted by samples in a batch. Columns are time values.
            int_quant_tensor (bool): If it is True, the save content will be converted to int format. False, save as original quant value.
            squeeze (bool): Whether to save one frame data in one cell in the table. And all the batches will be saved in one file.\
                            If it is False, every batch will be saved in its own file.\
                            It's not recommended set this as False in non-debug-situation if the input data contains time scale, \
                            or there will be lot of files.
                            
        """
        super().__init__()

        self.dirpath=dirpath
        self.filename=filename
        self.sort_by=sort_by
        self.int_quant_tensor=int_quant_tensor
        self.squeeze=squeeze

        self.batch_count=0

        # These parameter need to be change during forwarding by time process.
        # They will be automatically injected.
        # Details in snn.forward_pass.evein()
        self.with_time_scale=False  
        self.time_count=0
        self.last_one=False

        self.sav_en=False

        if os.path.exists(dirpath):
            if overwrite is True:
                print(f"Warning! Directory {dirpath} exists! Data could be overwritten.")   # .to_csv() will automatically overwrite file

            else:
                dirpath = dirpath + "_new"
                os.makedirs(dirpath)
                print(f"Input data will be saved in {dirpath}.")
        else:
            os.makedirs(dirpath)
                
        
    def forward(self, x):
        # x: tensor/QuantTensor, [batch_size, frame], with/without time scale.
        if self.sav_en is True:
            if isinstance(x, QuantTensor):
                if self.int_quant_tensor is True:
                    convert_x=float_quant_tensor2int(x).detach().cpu().tolist()
                else:
                    convert_x=x[0].detach().clone().cpu().tolist()
            elif isinstance(x, torch.Tensor):
                convert_x=x.detach().clone().cpu().tolist()
            elif isinstance(x, list):
                convert_x=x.copy()
            else:
                raise ValueError("Unknown input type!")
            

            # convert_x=[]
            # for x_item in x_int:
            #     if isinstance(x_item,QuantTensor):
            #         if self.int_quant_tensor is True:
            #             convert_x_item=snnu.float_quant_tensor2int(x_item).detach().cpu().tolist()
            #         else:
            #             convert_x_item=x_item[0].detach().clone().cpu().tolist()
            # # elif isinstance(x,torch.Tensor):
            # #     # convert_x=[list(map(list, x))]
            # #     convert_x=x.cpu()
            #     else:
            #         convert_x_item= x_item.detach().clone().cpu().tolist()
                
            #     convert_x.append(convert_x_item)


            if self.sort_by == "time" and (self.squeeze ): # Save by time dimension (line is time value) and one frame in a cell.
                df = pd.DataFrame({f"Sample{i}": [convert_x[i].tolist()] if isinstance(convert_x[i], torch.Tensor) else [convert_x[i]] for i in range(len(convert_x))})
            elif self.sort_by == "elem" and (self.squeeze ):  # Save by samples in a batch (line is samples) and one frame in a cell.
                df = pd.DataFrame({f"Time{self.time_count}": [row.tolist()] if isinstance(row, torch.Tensor) else [row] for row in convert_x})
            elif self.sort_by == "time" and not self.squeeze: # Save by time dimension (line is time value) and frame data spreed in cells.
                df = pd.DataFrame({f"Sample{i}": convert_x[i].tolist() if isinstance(convert_x[i], torch.Tensor) else convert_x[i] for i in range(len(convert_x))})
            elif self.sort_by == "elem" and not self.squeeze:  # Save by samples in a batch (line is samples) and frame data spreed in cells.
                df = pd.DataFrame({f"Time{self.time_count}": row.tolist() if isinstance(row, torch.Tensor) else row for row in convert_x})
            else:
                raise ValueError("Wrong set of `sort_by`.")
            
            if self.with_time_scale and self.squeeze:
                unit_str=f"batch_{self.batch_count}"
            elif self.with_time_scale is False and self.squeeze:
                unit_str="all_batches"
            elif self.with_time_scale and self.squeeze is False:
                unit_str=f"batch_{self.batch_count}_time_{self.time_count}"
            elif self.with_time_scale is False and self.squeeze is False:
                unit_str=f"batch_{self.batch_count}"
            
            if self.filename is None:
                fpath = os.path.join(self.dirpath, f"input_data_{unit_str}.csv")
            else:
                fpath = os.path.join(self.dirpath, f"{self.filename}_{unit_str}.csv")

            if self.squeeze is False:
                df.to_csv(fpath,header=True,index=False,)
                self.batch_count=self.batch_count+1 # Trying not to use inplace.
                # 
            else:
                if os.path.exists(fpath):
                    existing_df = pd.read_csv(fpath,dtype="str")
                    df = df.reindex(columns=existing_df.columns, fill_value="")
                    combined_df = pd.concat([existing_df, df],ignore_index=True)
                else:
                    combined_df=df
                combined_df.to_csv(fpath,header=True,index=False,)
                if self.last_one is True:
                # If you use `squeeze` with data contained no time scale, this branch will not activate.
                    self.batch_count=self.batch_count+1
            

        return x