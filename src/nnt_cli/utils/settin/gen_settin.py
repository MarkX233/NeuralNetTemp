import multiprocessing
import torch
import os
import re
import json
from torch.utils.data import random_split
from sqlalchemy import create_engine, text
from pathlib import Path
from urllib.parse import urlparse

def get_num_workers(mode,dist_num=4):
    """
    Allocate current CPU cores.

    Args:
        mode (str):
            Determine the current allocation mode.
            `"Full"` : Full use of avaliable CPU cores.
            `"Half"` : Use half of avaliable CPU cores.
            `"Auto"` : Automatically determine the number of cores currently in use, generally the number of GPUs * 4.
            `"Dist"` : Mode for distribute GPU. Return number of `cores / dist_num`.
    """
    try:
        cpu_count = multiprocessing.cpu_count()
        if mode == "Full":
            return cpu_count
        elif mode == "Auto":
            return min(cpu_count, torch.cuda.device_count() * 4)
        elif mode == "Half":
            return max(1, cpu_count // 2)
        elif mode == "Dist":
            return max(1, cpu_count // dist_num)
        else:
            raise ValueError("Wrong set of CPU allocate mode!")
    except Exception as e:
        print(f"Error detecting CPU count: {e}")
        return 0

# def get_notebook_name(no_extension=True):
#     """
#     Do not use it!
#     Only works in VS Code.
#     """
#     ip = get_ipython()
#     path = None
#     if '__vsc_ipynb_file__' in ip.user_ns:
#         path = ip.user_ns['__vsc_ipynb_file__']

#     file_name=os.path.basename(path)

#     if no_extension is True:
#         file_name=os.path.splitext(file_name)[0]

#     return file_name

def get_ipynb_files(directory,sub_dir=False):
    """Get all .ipynb files in the specified directory and its subdirectories"""
    ipynb_files = []

    exclude_notebook_name_pattern = re.compile(r".*_output.*")

    if sub_dir is True:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".ipynb") and not exclude_notebook_name_pattern.match(file.strip()):
                    ipynb_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if file.endswith(".ipynb") and not exclude_notebook_name_pattern.match(file.strip()):
                ipynb_files.append(os.path.join(directory, file))
    return ipynb_files


def get_notebook_name(dir_path, raise_error=True,sub_dir=False,get_path=False):
    """
    Get notebook name or path and update in source code.
    This function will search for all .ipynb files in the specified directory and its subdirectories,
    and check if the `get_notebook_name()` function is used in the code cells.
    If it is used, it will check if the notebook name is set correctly.
    If the notebook name is not set or incorrect, it will update the source code with the correct notebook name.
    If `get_path` is True, it will also check if the notebook path is set correctly.
    If `raise_error` is True, it will raise an error if any notebook name is not set or incorrect.
    Args:
        dir_path (str): The directory path to search for .ipynb files.
        raise_error (bool): Whether to raise an error if any notebook name is not set or incorrect.
        sub_dir (bool): Whether to search in subdirectories.
        get_path (bool): Whether to check and update the notebook path as well.
    """

    ipynb_files=get_ipynb_files(dir_path,sub_dir=sub_dir)

    get_notebook_name_pattern = re.compile(r".*get_notebook_name\s*\(.*\)")
    # .*: Match any number of any characters
    # \s*: Match zero or more whitespace characters
    # \(\) Match bracket

    error_found = False
    for file_path in ipynb_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        cells = content.get("cells", [])
        updated = False

        for cell in cells:
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                for i, line in enumerate(source):
                    
                    # if fnmatch(line.strip(),"*get_notebook_name=(*") and not fnmatch(line.strip(),"#*get_notebook_name=(*"):
                    # if fnmatch(line.strip(),"*get_notebook_name=(*") :
                    if get_notebook_name_pattern.match(line.strip()) and not line.strip().startswith("#"):
                        # Check if notebook_name=<name> already exists
                        if i + 1 < len(source) and source[i + 1].strip().startswith("notebook_name="):
                            notebook_line = source[i + 1].strip()
                            expected_notebook = os.path.basename(file_path).split(".ipynb")[0]
                            
                            # Check if the result is correct
                            if notebook_line == f"notebook_name='{expected_notebook}'":
                                print(f"[OK] {file_path}: Notebook name is correct.")
                            else:
                                print(f"[UPDATE] {file_path}: Incorrect notebook name. Updating...")
                                source[i + 1] = f"notebook_name='{expected_notebook}'\n"
                                updated = True
                                error_found = True
                        elif i + 2 < len(source) and get_path and source[i + 2].strip().startswith("notebook_path="):
                            notebook_path_line = source[i + 2].strip()
                            expected_notebook_path = os.path.abspath(file_path)
                            
                            # Check if the result is correct
                            if notebook_path_line == f"notebook_path='{expected_notebook_path}'":
                                print(f"[OK] {file_path}: Notebook path is correct.")
                            else:
                                print(f"[UPDATE] {file_path}: Incorrect notebook path. Updating...")
                                source[i + 2] = f"notebook_path='{expected_notebook_path}'\n"
                                updated = True
                                error_found = True
                        else:
                            # No results, insert notebook_name=<name>
                            print(f"[MISSING] {file_path}: Missing notebook name. Inserting...")
                            source.insert(i + 1, f"notebook_name='{os.path.basename(file_path).split('.ipynb')[0]}'\n")
                            updated = True
                            error_found = True
                    

                        # Update the source of cell
                        cell["source"] = source
                        break

        # If there are updates, save the file
        if updated:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)

    if raise_error is True and error_found is True:
        raise ValueError("No actuall error, but here is to remind you notebook name setting in source code is updated. \n \
            Close the IDE of the code tab (if there is a warning, choose revert not overwriten), if your notebook name hasn't been updated, \
                         and rerun the jupter notebook then it should work fine.")

    return error_found


def spilt_dataset(infer_size, dataset_in):
    """
    Spilt dataset for inference.
    """
    if infer_size >=1:
        infer_subset=dataset_in
        test_subset=dataset_in
    else:
        subset_out, infer_subset = random_split(dataset_in, [1-infer_size, infer_size])
    
    return subset_out, infer_subset


def ensure_db_exists(db_url: str, mk_db:bool = True) -> None:
    """Automatically create database files and directories (if not present)"""
    if db_url.startswith("sqlite:///"):
        db_path = Path(urlparse(db_url).path)
        db_dir = db_path.parent
        
        if not db_path.exists() and mk_db:
            # Create the database file if it doesn't exist
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path.touch(mode=0o664)
            print(f"Created new database: {db_path}")
            
            engine = create_engine(db_url, connect_args={"timeout": 30})
            with engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL;"))
                conn.execute(text("PRAGMA synchronous=NORMAL;"))
            engine.dispose()
            print(f"Configured database: {db_path}")
        elif not db_path.exists():
            print(f"Database does not exist and mk_db is False: {db_path}")
        else:
            print(f"Database already exists: {db_path}")
        


def cleanup_sqlite_locks(db_url):
    if db_url.startswith("sqlite:///"):
        parsed = urlparse(db_url)
        db_path = parsed.path
        for suffix in ['-journal', '-wal', '-shm']:
            lock_path = db_path + suffix
            if os.path.exists(lock_path):
                print(f"Deleting lock file: {lock_path}")
                os.remove(lock_path)
