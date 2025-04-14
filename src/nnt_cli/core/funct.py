from pathlib import Path
from importlib import resources
import os
import shutil
import sys
import subprocess
from datetime import datetime
import torch

import nnt_cli

from nnt_cli.core.core_utils import find_files, select_template_interactive, run_git, init_repo, OptTaskRunner, OptTaskRunner_static
from nnt_cli.core.gen_init import generate_package_init

nnt_path=Path(nnt_cli.__file__).parent

CUS_DIR= nnt_path / "custom"

CUS_TDIR = CUS_DIR / "custom_templates"
CUS_UDIR = CUS_DIR / "custom_utils"

global builtin_tpath
builtin_tpath = resources.files("nnt_cli.templates")

global builtin_upath
builtin_upath = resources.files("nnt_cli.utils")

def copy_files(args):

    search_paths = [CUS_TDIR, CUS_UDIR, builtin_tpath]

    found_files = find_files(args.file_name, search_paths)

    if not found_files:
        print(f"Error: Can't find files {args.file_name}")
        sys.exit(1)

    selected_path = (found_files[0] if len(found_files) == 1 
                    else select_template_interactive(found_files))

    try:
        dest = Path(args.target_dir).resolve() / args.file_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(selected_path, dest)
        print(f"Created successfully: {dest}")
        print(f"Source files: {selected_path}")

    except Exception as e:
        print(f"Copy failed: {str(e)}")
        sys.exit(1)

def del_files(args):
    """Delete files in custom directory"""
    search_paths = [CUS_TDIR, CUS_UDIR]

    found_files = find_files(args.file_name, search_paths)

    if not found_files:
        print(f"Error: Can't find template {args.file_name}")
        sys.exit(1)

    selected_path = (found_files[0] if len(found_files) == 1 
                    else select_template_interactive(found_files))
    try:
        
        os.remove(selected_path)
        print(f"Deleted successfully: {selected_path}")

    except Exception as e:
        print(f"Deleted failed: {str(e)}")
        sys.exit(1)

    generate_package_init(CUS_DIR,recursive=True)

def save_files(args):
    """Save template to global directory"""

    if args.utils is True:
        save_dir = CUS_UDIR
    else:
        save_dir = CUS_TDIR

    src = Path(args.file_path)
    dest = save_dir / src.name

    save_dir.mkdir(parents=True,exist_ok=True)

    if dest.exists() and not args.force:
        print(f"File {dest.name} already exists, use `--force` to overwrite it!")
        return
    try:
        shutil.copy(src, dest)
        print(f"File is saved at: {dest}")
    except Exception as e:
        print(f"Copy failed: {str(e)}")
        sys.exit(1)

    # Regenerate __init__.py file to add newly saved templates.
    generate_package_init(CUS_UDIR if args.utils else CUS_TDIR,recursive=True)

def list_files(args):
    """List all templates or custom utils"""
    # Internal
    builtin_templates = []
    for file in builtin_tpath.iterdir():
        if file.name != "__init__.py" and file.name != "__pycache__":
            builtin_templates.append(file.name)

    # Custom
    custom_templates = []
    if CUS_TDIR.exists():
        for file in CUS_TDIR.iterdir():
            if file.name != "__init__.py" and file.name != "__pycache__":
                custom_templates.append(file.name)
    
    custom_utils=[]
    if CUS_UDIR.exists():
        for file in CUS_UDIR.iterdir():
            if file.name != "__init__.py" and file.name != "__pycache__":
                custom_utils.append(file.name)

    print("Available Files:")

    print("=================Internal Template====================")
    for name in builtin_templates:
        tar_path=""
        if args.path is True:
            tar_path=builtin_tpath / name
        print(f" - {name}: [Internal Template] {tar_path}")

    print("=================Custom Template======================")    
    for name in custom_templates:
        tar_path=""
        if args.path is True:
            tar_path=CUS_TDIR / name
        print(f" - {name}: [Custom Template] {tar_path}")

    print("=================Custom Utils=========================")
    for name in custom_utils:
        tar_path=""
        if args.path is True:
            tar_path=CUS_UDIR / name
        print(f" - {name}: [Custom Utils] {tar_path}")

def create_project(args):
    """Create new project"""
    proj_name=str.capitalize(args.project_name)
    target_dir = Path(args.target_dir).resolve() / proj_name
    if target_dir.exists():
        print(f"Error: The directory already exists - {target_dir}")
        return
    
    if hasattr(args,"template") and args.template is not None:
        # Use indicated template
        template_path=find_files(args.template)
        temp_flag=True
    else:
        # Use internal template
        temp_flag=False
        template_path = builtin_tpath / "pro_temp.py"
        if not template_path.exists():
            print(f"Error: Not found default template - {template_path}")
            return

    target_dir.mkdir(parents=True)
    print(f"Created project directory: {target_dir}")

    rename_list=[]

    dest_file = target_dir / "pro_temp.py"
    shutil.copy(template_path, dest_file)
    print(f"Copy template file: {dest_file}")
    rename_list.append(dest_file)

    task_script_file=template_path =builtin_tpath / "task_script.py"
    shutil.copy(task_script_file, target_dir)
    print(f"Copy task script: {task_script_file}")
    rename_list.append(target_dir / "task_script.py")

    ipynb_file=template_path =builtin_tpath / "model.ipynb"
    shutil.copy(ipynb_file, target_dir)
    print(f"Copy model template: {ipynb_file}")
    rename_list.append(target_dir / "model.ipynb")


    for file in rename_list:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        if os.path.basename(file) == "pro_temp.py" and temp_flag:
            content = content.replace(args.template, proj_name)
        else:
            content = content.replace("_Project_", proj_name + "_")
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Rename the file - {file}")

def export_custom(args):
    export_dir = Path(args.export_dir).resolve()

    if not export_dir.exists():
        raise FileNotFoundError(f"Error: The target directory does not exist - {export_dir}")
    
    shutil.copytree(CUS_DIR, export_dir/"custom")
    print(f"Custom files are exported to {export_dir}")

def import_custom(args):
    import_dir = Path(args.import_dir).resolve()

    if import_dir.is_file():
        raise FileExistsError("Import object must be a folder! Use `save-file` to save file.")

    if not import_dir.exists():
        raise FileNotFoundError(f"Error: The target directory does not exist - {import_dir}")
    
    shutil.copytree(str(import_dir), str(CUS_DIR),ignore=shutil.ignore_patterns('.git', '*.git*'),dirs_exist_ok=True)
    print(f"Custom folder {import_dir} are imported to {CUS_DIR}.")

    generate_package_init(CUS_DIR,recursive=True)

    
def git_proxy(args):
    """Proxy all git commands to custom code directory"""
    custom_path = CUS_DIR
    
    custom_path.mkdir(exist_ok=True)
    
    # git_cmd = ["git", "-C", str(custom_path)] + args.git_args
    
    run_git(args.git_args,custom_path, save_log=False)

    
def sync_command(args):
    """One-step synchronization of custom code"""
    custom_path = CUS_DIR

    try:
        if args.init:
            print("First time to sync, initializing...")
            init_repo(custom_path)
            run_git(["branch","-M",args.branch],custom_path)

        else:
            print("Automatically submit local modifications...")
            run_git(["add", "."],custom_path)
            run_git(["commit", "-m", f"Auto commit: {datetime.now().isoformat()}"],custom_path)
            if not args.no_pull:
                print("Getting the latest version...")
                run_git(["pull", "--rebase", "origin", args.branch],custom_path)

            if not args.no_push:
                print("Push changes to remote repository...")
                run_git(["push", "origin", args.branch],custom_path)
            
            print(f"Synchronization is complete! Branches: {args.branch}")
        
    except subprocess.CalledProcessError as e:
        print(f"Synchronization failed: {e.stdout}. You can manually use `nnt git` to commit changes.")
        sys.exit(1)
    
    generate_package_init(CUS_DIR,recursive=True)


def opt_command(args):
    tasks = []
    task_args={"train_method": 'opt'} if args.trial <= -1 else {"train_method": 'opt', "n_trials": args.trial}
    thread=torch.cuda.device_count() if args.thread <= -1 else args.thread

    assert args.target.endswith(".ipynb"), "The target file must be a Jupyter notebook (.ipynb)"
    assert isinstance(args.static, list) and all(isinstance(x, int) for x in args.static), f"The static GPU list is not valid - {args.static}, it must be a list of integers."
    assert len(args.static) < thread and max(args.static) < thread, f"The static GPU list is not valid - {args.static}, it must be a list of integers less than the thread number - {thread}."

    if args.static is None:
        # Dynamic GPU allocation
        for th in range(thread):
            tasks.append((args.target, task_args))
        
        runner = OptTaskRunner(tasks, args.kernel, output2terminal=args.output, logfile=args.log)
    else:
        # Static GPU allocation
        for th in args.static:
            tasks.append((args.target, th, task_args))

        runner = OptTaskRunner_static(tasks, args.kernel, output2terminal=args.output, logfile=args.log)



