from pathlib import Path
import subprocess
from datetime import datetime
import sys
import os

from nnt_cli.utils.parallel_run.parallel_run import ParallelNotebookRunner
from nnt_cli.utils.parallel_run.parallel_run_dyn import DynamicAssignTaskOnGPU

def find_files(template_name, search_paths):
    """Search for template files in multiple directories"""
    found_files = []
    for path in search_paths:
        search_path = Path(path) / template_name
        if search_path.exists():
            found_files.append(search_path)
            
        # for f in Path(path).rglob(template_name):
        #     if f.is_file():
        #         found_files.append(f)
    if not found_files:
        raise FileNotFoundError(f"Can't find {template_name} in directories.")

    return found_files

def select_template_interactive(found_files):
    """Let the user choose when multiple templates of the same name are found"""
    print(f"Found {len(found_files)} templates of the same name:")
    for i, path in enumerate(found_files, 1):
        print(f"{i}. {path}")

    while True:
        try:
            choice = int(input("Please select the template to use (enter the number):"))
            if 1 <= choice <= len(found_files):
                return found_files[choice-1]
            print("Entering the number out of range, please re-enter")
        except ValueError:
            print("Please enter a valid number")

def run_git(cmd, custom_path, save_log=True, echo=True):
        """Execute Git commands and log logs"""
        log_file = custom_path / "sync.log"
        full_cmd = ["git", "-C", str(custom_path)] + cmd
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            result = subprocess.run(
                full_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            if echo:
                for line in result.stdout:
                    print(line, end='')
            
            if save_log:
                log = f"[{timestamp}] SUCCESS: {' '.join(full_cmd)}\n{result.stdout}\n"
            
            # result.wait()
            # sys.exit(result.returncode)

        except subprocess.CalledProcessError as e:
            if save_log:
                log = f"[{timestamp}] ERROR: {' '.join(full_cmd)}\n{e.stdout}\n"
            raise e
        
        except FileNotFoundError:
            print("Error: Git was not found, please install Git first")
            sys.exit(1)
        
        if save_log:
            with open(log_file, "a") as f:
                f.write(log + "\n")
        return result

def run_git_script(script_file,repo_path, branch):
    env = os.environ.copy()
    env.update({
        "REPO_PATH": repo_path,
        "Branch": branch
    })
    
    try:
        result = subprocess.run(
            [f"./{script_file}"],
            env=env,
            check=True,
            text=True,
            capture_output=True
        )
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)

GITIGNORE_CONTENT = """
*.pyc
__pycache__
sync.log
"""
def init_repo(custom_path):
    gitignore = custom_path / ".gitignore"
    if not gitignore.exists():
        print("Generating a new gitignore file.")
        gitignore.write_text(GITIGNORE_CONTENT)

class OptTaskRunner(DynamicAssignTaskOnGPU):
    def __init__(self, tasks, kernel,output2terminal=False, logfile=False):
        self.kernel = kernel
        self.output2terminal = output2terminal
        self.log = logfile
        self.notebooks_tasks = tasks
        self.util_threshold = 5
        self.memory_threshold = 30
        self.start_num= 1
        self.get_gpu_status()
        self.assign_and_exe()

class OptTaskRunner_static(ParallelNotebookRunner):
    def __init__(self, tasks, kernel,output2terminal=False, logfile=False):
        self.kernel = kernel
        self.output2terminal = output2terminal
        self.log = logfile
        self.notebooks_tasks = tasks

        self.run_tasks()