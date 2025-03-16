import os
import re
import torch
import pandas as pd
from json.decoder import JSONDecodeError
import json
import torch.nn as nn
import snntorch as snn
import brevitas as qnn

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
                "Test Accuracy": results[2],
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
    if one_time is True:
        flat_record = {
            "Title": filename,
            "Train Loss": record[0],
            "Train Accuracy": record[1],
            "Test Accuracy": record[2],
            "Infer Accuracy": record[3],
        }
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
        results (list): A list containing the results of the training and testing process. 
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

    final_record = {
        "Title": titlename,
        f"{vary_x_name}": vary_x,
        f"{vary_y_name}": vary_y,
        f"{vary_z_name}": vary_z,
        # "HPID": hpid,
        "Train Loss": results[0][-1],
        "Train Accuracy": results[1][-1],
        "Test Accuracy": results[2][-1],
        "Infer Accuracy": results[3][-1],
    }

    for key, value in final_record.items():
        if isinstance(value,list):
            for i in range(len(value)-1):
                eq=value[i+1]-value[i]
            
            if eq==0:
                final_record[key]=value[0]
            else:
                final_record[key]=str(value)
                

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
        test_acc = df["Test Accuracy"].tolist()
        infer_acc = df["Infer Accuracy"].tolist()
        title = df["Title"].tolist()

        x = df[xname].tolist()

        subtitle = f"x-Axis {xname}: {x}"

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
                    custom_xlabel=xname
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
                custom_xlabel=xname
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

    return [train_loss, train_acc, test_acc, infer_acc]


def load_record_and_plot(fpath):
    df = pd.read_csv(fpath)

    file_name = os.path.splitext(os.path.basename(fpath))[0]
    dir_path = os.path.dirname(fpath)

    i=0
    for i in range(len(df)):

        sup_title=df["Experiment"][i]

        try:
            train_loss = json.loads(df["Train Loss"][i])
        except JSONDecodeError:
            print(f"Warning! Can't load train loss value of {sup_title} in {fpath}. Proceeding to the next one.")
            continue
        try:
            train_acc = json.loads(df["Train Accuracy"][i])
        except JSONDecodeError:
            print(f"Warning! Can't load train accuracy value of {sup_title} in {fpath}. Proceeding to the next one.")
            continue
        try:
            test_acc = json.loads(df["Test Accuracy"][i])
        except JSONDecodeError:
            print(f"Warning! Can't load test accuracy value of {sup_title} in {fpath}. Proceeding to the next one.")
            continue
        try:
            infer_acc = json.loads(df["Infer Accuracy"][i])
        except JSONDecodeError:
            print(f"Warning! Can't load inference accuracy value of {sup_title} in {fpath}. Proceeding to the next one.")
            continue
        

        plot_acc(
            len(test_acc),
            train_loss,
            train_acc,
            test_acc,
            infer_acc,
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

def get_next_demo_index(directory, match_word, file_extension=".csv"):
    """
    Find the next available index for files named '`match_word`*.<`file_extension`>' in the specified directory.

    Args:
        directory (str): The path to the directory to check.
        match_word (str): The word to match in the filenames or sub directory names.
        file_extension (str): The file extension to look for. Defaults to ".csv".
            If it is "dir", match pattern is for directories.

    Returns:
        int: The next available index for a new file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 1

    # List all files in the directory
    items = os.listdir(directory)

    if file_extension =="dir":
        demo_pattern = re.compile(re.escape(match_word) + r"(\d+)")
    else:    
        demo_pattern = re.compile(re.escape(match_word) + r"(\d+)" + re.escape(file_extension))
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
                float_quant_tensor2int(layer.quant_weight()).flatten().detach().cpu().tolist()
            )
            if layer.bias is not None:
                layer_dict[f"QuantLinear{lin}_bias"] = (
                    float_quant_tensor2int(layer.quant_bias()).flatten().detach().cpu().tolist()
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


def save_checkpoint(model, optimizer, epoch, loss, train_l_list, train_acc_list, test_acc_list, infer_acc_list, path="./checkpoint", scheduler=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'train_l_list': train_l_list,
        'train_acc_list': train_acc_list,
        'test_acc_list': test_acc_list,
        'infer_acc_list': infer_acc_list
    }

    if scheduler is not None:
        checkpoint["scheduler_state"]=scheduler.state_dict()

    N = 2
    # Only retain the recent N CHECKPOINT

    os.makedirs(path, exist_ok=True)

    torch.save(checkpoint, f"{path}/checkpoint_{epoch}.pth")
    print(f"Checkpoint saved at epoch {epoch}")

    checkpoint_files = sorted(
        [f for f in os.listdir(path) if f.startswith("checkpoint_")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    if len(checkpoint_files) > N:
        os.remove(os.path.join(path, checkpoint_files[0]))
        # Delete the oldest file

def load_checkpoint(model, optimizer, fpath="checkpoint.pth", scheduler=None):
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler_state = checkpoint.get("scheduler_state", None)

    if scheduler_state is not None and scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    else:
        print("Can't find scheduler_state.")

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    train_l_list = checkpoint.get('train_l_list', [])
    train_acc_list = checkpoint.get('train_acc_list', [])
    test_acc_list = checkpoint.get('test_acc_list', [])
    infer_acc_list = checkpoint.get('infer_acc_list', [])

    
    
    print(f"Checkpoint loaded from epoch {epoch}")
    return loss, [epoch, train_l_list, train_acc_list, test_acc_list, infer_acc_list]
