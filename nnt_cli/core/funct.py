from pathlib import Path
from importlib import resources
import pkgutil
import os
import shutil
import sys

from .core_utils import find_files, select_template_interactive
from .gen_init import generate_package_init

global custom_tpath
custom_tpath = Path.home() / ".nnt_cli" / "custom_templates"

global custom_upath

custom_upath = Path.home() / ".nnt_cli" / "custom_utils"

global builtin_tpath
builtin_tpath = Path(__file__).parent / "templates"

global builtin_upath
builtin_upath = Path(__file__).parent / "utils"

def copy_template(args):

    search_paths = [custom_tpath, builtin_tpath]

    found_files = find_files(args.template_name, search_paths)

    if not found_files:
        print(f"Error: Can't find template {args.template_name}")
        sys.exit(1)

    selected_path = (found_files[0] if len(found_files) == 1 
                    else select_template_interactive(found_files))

    try:
        dest = Path(args.target_dir).resolve() / args.template_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(selected_path, dest)
        print(f"Created successfully: {dest}")
        print(f"Source template: {selected_path}")

    except Exception as e:
        print(f"Copy failed: {str(e)}")
        sys.exit(1)

def save_files(args):
    """Save template to global directory"""

    if args.utils is True:
        save_dir = custom_upath
    else:
        save_dir = custom_tpath

    src = Path(args.file_path)
    dest = save_dir / src.name

    save_dir.mkdir(parents=True,exist_ok=True)

    if dest.exists() and not args.force:
        print(f"Template {dest.name} already exists, use `--force` to overwrite it!")
        return
    try:
        shutil.copy(src, dest)
        print(f"Template is saved at: {dest}")
    except Exception as e:
        print(f"Copy failed: {str(e)}")
        sys.exit(1)

    # Regenerate __init__.py file to add newly saved templates.
    generate_package_init("nnt_cli",recursive=True)

def list_files(args):
    """List all templates or custom utils"""
    # Internal
    builtin_templates = []
    for module_info in pkgutil.iter_modules([str(builtin_tpath)]):
        if module_info.name != "__init__":
            builtin_templates.append(module_info.name)

    # Custom
    custom_templates = []
    if custom_tpath.exists():
        for file in custom_tpath.glob("*"):
            if file.stem != "__init__":
                custom_templates.append(file.stem)
    
    custom_utils=[]
    if custom_upath.exists():
        for py_file in custom_upath.glob("*.py"):
            if py_file.stem != "__init__":
                custom_utils.append(py_file.stem)

    # Merge and deduplicate (custom priority)
    all_templates = {}

    for name in builtin_templates:
        all_templates[name] = {"loc":"Internal"}
            
    for name in custom_templates:
        all_templates[name] = {"loc":"Custom"}
    
    for name in custom_utils:
        all_templates[name] = {"loc":"Custom Utils"}

    print("Available Files:")
    for name, tar_dict in sorted(all_templates.items()):
        tar_path=""
        if args.path is True:
            if tar_dict["loc"] == "Internal":
                tar_path=builtin_tpath / name
            elif tar_dict["loc"] == "Custom":
                tar_path=custom_tpath / name
            elif tar_dict["loc"] == "Custom Utils":
                tar_path=custom_upath / name
        print(f" - {name}: [{tar_dict['loc']}] {tar_path}")

def create_project(args):
    """Create new project"""
    target_dir = Path(args.dir).resolve() / args.project_name
    if target_dir.exists():
        print(f"Error: The directory already exists - {target_dir}")
        return
    
    if args.template:
        # Use indicated template

        template_path=find_files(args.template)

    else:
        # Use internal template
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
    rename_list.append(task_script_file)

    ipynb_file=template_path =builtin_tpath / "model.ipynb"
    shutil.copy(ipynb_file, target_dir)
    print(f"Copy model template: {ipynb_file}")
    rename_list.append(ipynb_file)


    for file in rename_list:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        if os.path.basename(file) == "pro_temp.py" and args.template:
            content = content.replace(args.template, args.project_name)
        else:
            content = content.replace("_Project_", args.project_name + "_")
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Rename the project - {args.project_name}")
