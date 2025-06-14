import os
import random
import re
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import snntorch as snn
import brevitas.nn as qnn
import ast

from nnt_cli.utils.plot.acc_plot import plot_acc
from nnt_cli.utils.data_trans import float_quant_tensor2int

def store_record(record, filename, one_time=False, path="./rec",HPID_en=True):
    """
    Store iteration's record of results.

    Args:
        record (dict | list):
            If `one_time` is false, it is dict of results:
            Example::

                record[f"{hparams['suptitle']}: {x}"]={
                "HPID": hpid,
                "Train Loss": results[0],
                "Train Accuracy": results[1],
                "Validation Accuracy": results[2],
                "Infer Accuracy": results[3],

            If `one_time` is true, record is a list of results
        }
        filename (str):
            Filename of whole record.
            filename not path.
        one_time (bool):
            If True, only store one result of training
        HPID_en (bool): 
            Data contains HPID.

    """
    file_path = f"{path}/{filename}.csv"

    if not os.path.exists(path):
        os.makedirs(path)

    flat_record = []
    if one_time is True and isinstance(record, list):
        flat_record = {
            "Title": filename,
            "Train Loss": record[0],
            "Train Accuracy": record[1],
            "Validation Accuracy": record[2],
            "Infer Accuracy": record[3],
        }
        if len(record) > 4:
            flat_record["Test Accuracy"] = record[4]
    else:
        for key, value in record.items():
            flat_row = {"Experiment": key}
            flat_row.update(value)
            flat_record.append(flat_row)

    df = pd.DataFrame(flat_record)

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, df])
        if one_time is False and HPID_en is True:
            combined_df = combined_df.drop_duplicates(subset="HPID", keep="last")
        # Same HPID record will be overwroten.
    else:
        combined_df = df

    if one_time is False:
        combined_df.sort_values(by="Experiment", ascending=True, inplace=True)
    combined_df.to_csv(file_path, index=False)
    print(f"Data of {filename} is saved!")


def store_final_results(
        results, titlename, vary_x=None, vary_y=None, hpid=None, single_file=True, total_file=True,
        total_name="default", vary_x_name="Varialbe Value", vary_y_name="Varialbe Value2", path="./rec",
        vary_z=None, vary_z_name="Varialbe Value3"
    ):
    """
    Stores the final results of a model's training and testing process into CSV files.
    Here provides 3 addtional variable sets besides 4 results, that you can add in the final results.
    Normally addtional variable set is used to assist plotting.

    Args:
        results (list): A list or dict containing the results of the training and testing process. 
                        Expected to be in the format [train_loss, train_accuracy, test_accuracy, infer_accuracy].
        titlename (str): The title name to be used for the single result file.
        vary_x (optional): The value of the variable 1.
        vary_y (optional): The value of the variable 2.
        vary_z (optional): The value of the variable 3.
        hpid (optional): Hyperparameter ID to uniquely identify the set of hyperparameters used.
        single_file (bool, optional): Whether to store the results in a single file specific to the title. Default is True.
        total_file (bool, optional): Whether to store the results in a total results file. Default is True.
        total_name (str, optional): The name to be used for the total results file. Default is "default".
        vary_x_name (str, optional): The name of the variable. Default is "Varialbe Value".
        vary_y_name (str, optional): The name of the variable. Default is "Varialbe Value2".
        vary_z_name (str, optional): The name of the variable. Default is "Varialbe Value3".
    
    Returns:
        None
    
    The function saves the final results into two CSV files:
    1. A single result file named as `rec/{titlename}_finals.csv`.
    2. A total results file named as `rec/{total_name}_total_results.csv`.
    
    If the files already exist, the function appends the new results to the existing files 
    and removes any duplicate entries based on the HPID (If exists).

    """

    if isinstance(results, list):
        final_record = {
            "Title": titlename,
            f"{vary_x_name}": vary_x,
            f"{vary_y_name}": vary_y,
            f"{vary_z_name}": vary_z,
            # "HPID": hpid,
            "Train Loss": results[0][-1],
            "Train Accuracy": results[1][-1],
            "Validation Accuracy": results[2][-1],
            "Infer Accuracy": results[3][-1],
        }

        if len(results) > 4:
            
            if isinstance(results[4],list):
                last_test_acc=results[4][-1]
            # elif isinstance(results[4], (int, float, torch.Tensor,)):
            else:
                last_test_acc=results[4]

            final_record["Test Accuracy"]=last_test_acc
            
    elif isinstance(results,dict):
        final_record = {
            "Title": titlename,
            f"{vary_x_name}": vary_x,
            f"{vary_y_name}": vary_y,
            f"{vary_z_name}": vary_z,
            # "HPID": hpid,

        }

        for key, value in results.items():
            if isinstance(value, list):
                final_record[key]=value[-1]
            else:
                final_record[key]=value

    for key, value in final_record.items():
        if isinstance(value, list):
            if all(x == value[0] for x in value):
                final_record[key] = value[0]
            else:
                final_record[key] = str(value)
                

    if hpid is not None:
        final_record["HPID"]=hpid

    tfile_path = f"{path}/{total_name}_total_results.csv"

    tdf = pd.DataFrame(final_record, index=[0])

    sinfile_path = f"{path}/{titlename}_finals.csv"

    if not os.path.exists(path):
        os.makedirs(path)

    if single_file is True:

        if os.path.exists(sinfile_path):
            try:
                existing_tdf = pd.read_csv(sinfile_path)
                combined_tdf = pd.concat([existing_tdf, tdf])
                if hpid is not None:
                    combined_tdf = combined_tdf.drop_duplicates(subset="HPID", keep="last")
            except (pd.errors.EmptyDataError, ValueError):
                print(
                    f"Warning: Can't add final result into {titlename} into {sinfile_path}!"
                )
        else:
            combined_tdf = tdf

        combined_tdf.sort_values(by="Title", ascending=True, inplace=True)
        combined_tdf.to_csv(sinfile_path, index=False)
        print(f"Data of final results of {titlename} is saved!")

    if total_file is True:

        if os.path.exists(tfile_path):
            try:
                existing_tdf = pd.read_csv(tfile_path)
                combined_tdf = pd.concat([existing_tdf, tdf])
                if hpid is not None:
                    combined_tdf = combined_tdf.drop_duplicates(subset="HPID", keep="last")
                # Same HPID record will be overwroten.
            except (pd.errors.EmptyDataError, ValueError):
                print(
                    f"Warning: Can't add final result {titlename} into {tfile_path}!"
                )
        else:
            combined_tdf = tdf

        combined_tdf.sort_values(by="Title", ascending=True, inplace=True)
        combined_tdf.to_csv(tfile_path, index=False)
        print(
            f"Data of final results of {titlename} is saved into total results file!"
        )

def load_final_csv_and_plot(fpath,xname,yname=None):
    """
    Load all the final results from a CSV file and plot them in a variable type scale.

    Corresponding to `store_final_results`

    Args:        
        fpath (str): The file path to the CSV file containing the results.
        xname (str, optional): The name of the column to be used for the x-axis. This name must be contained in the result csv file.
        yname (str, optional): Second variable name which will be showed in title. Used to display the results that varys by 2 variables. \
                               If None, a default plot that only varys by x is generated.\
                               `yname` must be contained in the result csv file.
    
    Returns:
        None
    """
    try:
        df = pd.read_csv(fpath)

        file_name = os.path.splitext(os.path.basename(fpath))[0]
        dir_path = os.path.dirname(fpath)

        train_loss = df["Train Loss"].tolist()
        train_acc = df["Train Accuracy"].tolist()
        vali_acc = df["Validation Accuracy"].tolist()
        infer_acc = df["Infer Accuracy"].tolist()
        title = df["Title"].tolist()

        if "Test Accuracy" in df.columns:
            test_acc = df["Test Accuracy"].tolist()
        else:
            test_acc = vali_acc
            # If there is no test accuracy, use validation accuracy.

        x = df[xname].tolist()

        subtitle = f"x-Axis {xname}: {x}"

        cus_title = ['Loss', 'Train Accuracy', 'Test Accuracy', 'Inference Accuracy']

        if isinstance(x[0], str):
            x = range(1, len(x) + 1)

        if yname is not None:
            y=df[yname].tolist()

            y_indices = {}
            for index, value in enumerate(y):
                if value not in y_indices:
                    y_indices[value]=[]
                y_indices[value].append(index)

            # print(y_indices)

            for y_value, indices in y_indices.items():
                
                train_loss_y=[train_loss[i] for i in indices]
                train_acc_y = [train_acc[i] for i in indices]
                test_acc_y = [test_acc[i] for i in indices]
                infer_acc_y = [infer_acc[i] for i in indices]
                x_y = [x[i] for i in indices]

                subtitle = f"x-Axis {xname}: {x_y}"

                

                plot_acc(
                    0,
                    train_loss_y,
                    train_acc_y,
                    test_acc_y,
                    infer_acc_y,
                    f"{file_name}-{yname}:{y_value}",
                    xlist=x_y,
                    sub_title=subtitle,
                    store_path=dir_path,
                    custom_xlabel=xname,
                    cus_title=cus_title
                )
        else:    
            plot_acc(
                0,
                train_loss,
                train_acc,
                test_acc,
                infer_acc,
                f"{file_name}_compare",
                xlist=x,
                sub_title=subtitle,
                store_path=dir_path,
                custom_xlabel=xname,
                cus_title=cus_title
                )
    except:
        raise RuntimeError("Can't plot the final results!")


def load_sin_res(fpath):
    """
    Load one single training's results.
    """
    df = pd.read_csv(fpath)

    train_loss = df["Train Loss"].tolist()
    train_acc = df["Train Accuracy"].tolist()
    test_acc = df["Test Accuracy"].tolist()
    infer_acc = df["Infer Accuracy"].tolist()
    
    if 'Validation Accuracy' in df.columns:
        vali_acc = df["Validation Accuracy"].tolist()
        return [train_loss, train_acc, vali_acc, infer_acc, test_acc]
    else:
        # Old version of results, without validation accuracy.
        print("Warning! No validation accuracy in the results file.")
        return [train_loss, train_acc, test_acc, infer_acc]


def load_record_and_plot(fpath):
    df = pd.read_csv(fpath)

    file_name = os.path.splitext(os.path.basename(fpath))[0]
    dir_path = os.path.dirname(fpath)

    plot_key=[
        "Train Loss",
        "Train Accuracy",
        "Test Accuracy",
        "Infer Accuracy",
        "Validation Accuracy"
    ]

    valid_key=[]
    for key in plot_key:
        if key in df.columns:
            valid_key.append(key)
        else:
            print(f"Warnings, can't load results of {key}.")

    plot_results={}

    for i in range(len(df)):
        sup_title=df["Experiment"][i]

        for key in valid_key:
            if isinstance(df[key][i], str):
                plot_results[key]=ast.literal_eval(df[key][i])
            else:
                plot_results[key]=df[key][i]

        if "Validation Accuracy" in plot_results:

            cus_title = ['Loss', 'Train Accuracy', 'Validation Accuracy', 'Evaluation Accuracy']

            if isinstance(plot_results['Test Accuracy'], list):
                test_acc=plot_results['Test Accuracy'][-1]
            else:
                test_acc=plot_results['Test Accuracy']

            if isinstance(plot_results['Infer Accuracy'], list):
                infer_acc=plot_results['Infer Accuracy'][-1]
            else:
                infer_acc=plot_results['Infer Accuracy']


            text_in_sub_4 = [f"Test Accuracy: {test_acc:.6f}",
                                f"Inference Accuracy: {infer_acc:.6f}"]

            plot_acc(
                len(plot_results['Train Loss']),
                plot_results['Train Loss'],
                plot_results['Train Accuracy'],
                plot_results['Validation Accuracy'],
                0,
                suptitle=sup_title,
                sub_title=file_name,
                store_path=dir_path,
                cus_title=cus_title,
                text_in_sub_4=text_in_sub_4
            )
        
        else:
            # Old version of results, without validation accuracy.
            plot_acc(
                len(plot_results['Train Loss']),
                plot_results['Train Loss'],
                plot_results['Train Accuracy'],
                plot_results['Test Accuracy'],
                plot_results['Infer Accuracy'],
                suptitle=sup_title,
                sub_title=file_name,
                store_path=dir_path
            )

def load_content(fpath,dic_name):
    """
    Load content from .csv.
    """

    df = pd.read_csv(fpath)

    return df[dic_name]

def get_next_demo_index(directory, match_word, file_extension=".csv", strict=True):
    """
    Find the next available index for files named '`match_word`*.<`file_extension`>' in the specified directory.

    Args:
        directory (str): The path to the directory to check.
        match_word (str): The word to match in the filenames or sub directory names.
        file_extension (str): The file extension to look for. Defaults to ".csv".
            If it is "dir", match pattern is for directories.
        strict (bool): If True, only match files with the exact name. Defaults to True. \
                        If False, match files with the name containing `match_word` and index is f'_{number}' at the last.

    Returns:
        int: The next available index for a new file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 1

    # List all files in the directory
    items = os.listdir(directory)

    index_pattern= r"(\d+)" if strict else r".*_(\d+)$"

    if file_extension =="dir":
        demo_pattern = re.compile(re.escape(match_word) + index_pattern)
    else:    
        demo_pattern = re.compile(re.escape(match_word) + index_pattern + re.escape(file_extension))
    indices = []

    for item in items:
        full_path = os.path.join(directory, item)
        
        # Check if we're matching directories or files
        if file_extension == "dir":
            if os.path.isdir(full_path):  # Only process directories
                match = demo_pattern.match(item)
                if match:
                    indices.append(int(match.group(1)))  # Extract the numeric index
        else:
            if os.path.isfile(full_path):  # Only process files
                match = demo_pattern.match(item)
                if match:
                    indices.append(int(match.group(1)))  # Extract the numeric index

    # If no matching items are found, start with index 1
    if not indices:
        return 1

    # Return the next index after the current maximum
    return max(indices) + 1

def sav_lin_net_paras(
    net, path, dname, index=False, learn_beta=False, learn_threshold=False, overwrite=False
):
    """
    Save network's parameters

    Using pandas.DataFrame, the parameters can be saved as .csv file.

    If using `learn_beta` and `learn_threshold`, the parameters will be saved as separate files.

    """

    lin = 0
    lea = 0
    layer_dict = {}

    dirpath = os.path.join(path, dname)

    if os.path.exists(dirpath):
        if overwrite is False:
            u_input = (
                input(f"Warning! Directory {dirpath} exists! Whether to delete files? (y/n)")
                .strip()
                .lower()
            )
            if u_input == "y":
                for filename in os.listdir(dirpath):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")
                print(f"Parameters will be saved in {dirpath}.")
            if u_input == "n":
                dirpath = dirpath + "_new"
                os.makedirs(dirpath)
                print(f"Parameters will be saved in {dirpath}.")
        else:
            print(f"Parameters will be saved in {dirpath}.")
    else:
        os.makedirs(dirpath)

    for _, layer in net.named_modules():
        if isinstance(layer, nn.Linear) and not isinstance(layer, qnn.QuantLinear):
        # Quantizied linear layer from brevitas will stil be consider as nn.Linear
            layer_dict[f"Linear{lin}_weight"] = (
                layer.weight.flatten().detach().cpu().tolist()
            )
            if layer.bias is not None:
                layer_dict[f"Linear{lin}_bias"] = (
                    layer.bias.flatten().detach().cpu().tolist()
                )
            lin = lin+1
        
        if isinstance(layer, qnn.QuantLinear):
            layer_dict[f"QuantLinear{lin}_weight"] = (
                float_quant_tensor2int(layer.quant_weight(),int_datatype='int',round=True).flatten().detach().cpu().tolist()
                # You can also use other float2int functions form brevitas, such as:
                # float_quant_tensor2int2(layer.quant_weight()).flatten().detach().cpu().tolist()
                # float_quant_tensor2int3(layer.quant_weight()).flatten().detach().cpu().tolist()
            )
            if layer.bias is not None:
                layer_dict[f"QuantLinear{lin}_bias"] = (
                    float_quant_tensor2int(layer.quant_bias(),int_datatype='int',round=True).flatten().detach().cpu().tolist()
                )
            # Normally when using QuantLinear, there will be no Linear Layer.
            lin = lin+1

        if isinstance(layer, snn.Leaky):
            if learn_beta is True:
                layer_dict[f"Leaky{lea}_beta"] = layer.beta.detach().cpu().tolist()
            if learn_threshold is True:
                layer_dict[f"Leaky{lea}_threshold"] = (
                    layer.threshold.detach().cpu().tolist()
                )
            if learn_threshold is False and learn_beta is False:
                layer_dict[f"Leaky{lea}"] = {
                    "Beta": layer.beta.detach().cpu().tolist(),
                    "Reset mechanism": layer.reset_mechanism_val.detach()
                    .cpu()
                    .tolist(),
                    "Threshold": layer.threshold.detach().cpu().tolist(),
                }
            lea = lea + 1

    for key, value in layer_dict.items():
        if isinstance(value, dict):
            df = pd.DataFrame.from_dict(value, orient="index")
        else:
            df = pd.DataFrame(value)
        # Save values in dict using the name of the key
        fpath = os.path.join(dirpath, f"{key}.csv")
        df.to_csv(fpath, index=index, header=False)
        print(f"{key} are saved in {fpath}.")

    print("All parameters are saved.")


def sav_net_state(net, path, dname, optimizer=None, loss=None, scheduler=None, overwrite=True):
    """
    Save network's state for further training or inference.
    """

    dirpath = os.path.join(path, dname)

    if os.path.exists(dirpath):
        # u_input = (
        #     input(f"Warning! Directory {dirpath} exists! Whether to overwrite? (y/n)")
        #     .strip()
        #     .lower()
        # )
        # if u_input == "y":
        #     for filename in os.listdir(dirpath):
        #         file_path = os.path.join(dirpath, filename)
        #         try:
        #             if os.path.isfile(file_path):
        #                 os.remove(file_path)
        #                 print(f"Deleted file: {file_path}")
        #         except Exception as e:
        #             print(f"Error deleting file {file_path}: {e}")
        #     print(f"Parameters will be saved in {dirpath}.")
        # if u_input == "n":
        #     dirpath = dirpath + "_new"
        #     os.makedirs(dirpath)
        #     print(f"Parameters will be saved in {dirpath}.")
        if overwrite is True:
            # Do nothing, because save function will overwrite by default.
            pass
        else:
            dirpath = dirpath + "_new"
            os.makedirs(dirpath)
            print(f"Parameters will be saved in {dirpath}.")
    else:
        os.makedirs(dirpath)

    torch.save(net.state_dict(), f"{dirpath}/net_params.pth")
    print(f"Network state is saved in {dirpath}/net_params.pth")

    if optimizer is not None:
        torch.save(optimizer.state_dict(), f"{dirpath}/optimizer_state.pth")
        print(f"Optimizer state is saved in {dirpath}/optimizer_state.pth")

    if loss is not None:
        torch.save(loss, f"{dirpath}/loss.pth")
        print(f"Loss state is saved in {dirpath}/loss.pth")
    
    if scheduler is not None:
        torch.save(scheduler, f"{dirpath}/scheduler_state.pth")
        print(f"Scheduler state is saved in {dirpath}/scheduler_state.pth")

    print("All parameters are saved.")


def load_net_state(net, path, optimizer=False, loss=False):
    """
    Load network's state for further training or inference.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Directory {path} does not exist. Check the path and directory name!"
        )

    net.load_state_dict(torch.load(f"{path}/net_params.pth"))
    print(f"Network state is loaded from {path}/net_params.pth")

    if optimizer is not False:
        optimizer.load_state_dict(torch.load(f"{path}/optimizer_state.pth"))
        print(f"Optimizer state is loaded from {path}/optimizer_state.pth")

    if loss is not False:
        loss=torch.load(f"{path}/loss_state.pth")
        print(f"Loss state is loaded from {path}/loss_state.pth")

    print("All parameters are loaded.")


CHECKPOINT_VERSION = 0.3
"""
Checkpoint version 0.3
- Added saving random state of numpy, python, torch and cuda.

Checkpoint version 0.2
- Added saving additional dict in checkpoint, for resuming iteration running.

Checkpoint version: 0.1
- Added checkpoint version control.
- Added saving `num_epochs`, initial number of epochs to run.

"""

def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    train_l_list,
    train_acc_list,
    test_acc_list,
    infer_acc_list,
    path="./checkpoint",
    scheduler=None,
    cp_retain=2, #  number of checkpoints to retain
    num_epochs=None,
    add_dict=None
):
    """
    Save the model checkpoint.
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The current loss value.
        train_l_list (list): List of training losses.
        train_acc_list (list): List of training accuracies.
        test_acc_list (list): List of testing accuracies.
        infer_acc_list (list): List of inference accuracies.
        path (str, optional): Directory to save the checkpoint. Defaults to "./checkpoint".
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler to save. Defaults to None.
        cp_retain (int, optional): Number of checkpoints to retain. Defaults to 2.
        add_dict (dict, optional): Additional dictionary to save in the checkpoint. Defaults to None.
    """

    checkpoint = {
        'checkpoint_version': CHECKPOINT_VERSION,
        'epoch': epoch,
        'num_epochs' : num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'train_l_list': train_l_list,
        'train_acc_list': train_acc_list,
        'test_acc_list': test_acc_list,
        'infer_acc_list': infer_acc_list,
        'random_state': torch.get_rng_state(),
        'cuda_random_state': torch.cuda.get_rng_state_all(),
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
    }

    if add_dict is not None:
        checkpoint.update(add_dict)
        # Add additional dict to checkpoint

    if scheduler is not None:
        checkpoint["scheduler_state"]=scheduler.state_dict()

    
    os.makedirs(path, exist_ok=True)

    torch.save(checkpoint, f"{path}/checkpoint_{epoch}.pth")
    print(f"Checkpoint saved at epoch {epoch}")

    checkpoint_files = sorted(
        [f for f in os.listdir(path) if f.startswith("checkpoint_")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    if len(checkpoint_files) > cp_retain:
        os.remove(os.path.join(path, checkpoint_files[0]))
        # Delete the oldest file
    

def load_checkpoint(fpath, model=None, optimizer=None, scheduler=None, device='cpu'):
    """
    Load a checkpoint from a file. And update the state dict of model, optimizer and scheduler.
    Args:
        fpath (str): Path to the checkpoint file.
        model (torch.nn.Module, optional): Model to load the state dict into. Defaults to None.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load the state dict into. Defaults to None.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load the state dict into. Defaults to None.
    Returns:
        checkpoint (dict): The loaded checkpoint.
    """
    checkpoint = torch.load(fpath,map_location=device)
    if model is not None:
        model = model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # for param in model.parameters():
        #     param = param.to(device)

        for module in model.modules():
            move_module_tensors_to_device(module, device)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            for key in param_group:
                if isinstance(param_group[key], torch.Tensor):
                    param_group[key] = param_group[key].to(device)
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    scheduler_state = checkpoint.get("scheduler_state", None)

    if scheduler_state is not None and scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
        for key, value in scheduler.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(scheduler, key, value.to(device))
        
    elif scheduler_state is None:
        print("Can't find scheduler state.")

    print(f"Checkpoint loaded from {fpath}")

    cp_version = checkpoint.get('checkpoint_version',None)
    print(f"Checkpoint version: {cp_version}")
    if cp_version is None:
        print("Warning! Checkpoint version is None. Checkpoint file could be an initial old version." \
        "Or it is not a checkpoint file.")
    elif cp_version < CHECKPOINT_VERSION:
        print("Warning! Checkpoint version is lower than current version. ")
    elif cp_version > CHECKPOINT_VERSION:
        print("Warning! Checkpoint version is higher than current version. Maybe you should update your code.")
    
    # Load random state
    if 'random_state' in checkpoint:
        torch.set_rng_state(checkpoint['random_state'])
    if 'cuda_random_state' in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])
    if 'python_random_state' in checkpoint:
        random.setstate(checkpoint['python_random_state'])
    if 'numpy_random_state' in checkpoint:
        np.random.set_state(checkpoint['numpy_random_state'])

    
    return checkpoint


def move_module_tensors_to_device(module, device):
    for name, value in vars(module).items():
        if isinstance(value, torch.Tensor):
            setattr(module, name, value.to(device))

def load_csv_params(fpath):
    """
    Load csv params file.
    """
    try:
        df = pd.read_csv(fpath, header=None)
        data = df[0].tolist()
        return data
    except FileNotFoundError:
        print(f"File {fpath} not found.")
        return None

def sav_csv_params(data, fpath):
    """
    Save params to csv file.
    """
    try:
        df = pd.DataFrame(data)
        df.to_csv(fpath, index=False, header=False)
        print(f"Data saved to {fpath}")
    except Exception as e:
        print(f"Error saving data to {fpath}: {e}")