import subprocess
import threading
import os
import argparse
import time
from datetime import datetime
from threading import Event
import sys

from nnt_cli.utils.settin.gen_settin import get_notebook_name
from nnt_cli.utils.sl import get_next_demo_index

class DynamicAssignTaskOnGPU():
    def __init__(self):
        """
        Dynamically assign tasks in the task list to run on idle GPU/GPUs.
        Both Linux and Windows are available.
        Before running, please inherit this class and override the `set_task` method.
        Then call the newly written class.
        """
        self.save_to_target = False
        self.cwd = os.getcwd()
        self.get_args()
        self.set_task()
        get_notebook_name("./",raise_error=False,sub_dir=True) # To start the kernel
        self.assign_and_exe()

    def set_task(self):
        """
        notebooks_and_params (list): A list of tuples where each tuple contains:
            - notebook (str): The name of the notebook file to run.
            - parameters (dict): A dictionary of parameters to pass to the notebook.
                                The parameters need to be first set in the certain cell with `parameters` tag in the notebook.
        """
        # self.save_to_target = False

        self.notebooks_tasks = [
            ("L2.ipynb", {"para_mode": True, "num_epochs" : 5}),
            ("L3.ipynb", {"para_mode": True, "num_epochs" : 5}),
            ("L4.ipynb", {"para_mode": True, "num_epochs" : 5}),
        ]

    # def sys_append(self):
    #     """
    #     Append the current directory to sys.path to ensure that the script can find the necessary modules.
    #     """
    #     if self.cross_dir_mode:
    #         for file in self.notebooks_tasks:
    #             if os.path.exists(file[0]):
    #                 current_dir = os.path.dirname(os.path.abspath(file[0]))
    #                 if current_dir not in sys.path:
    #                     sys.path.append(current_dir)


    
    def get_args(self):
        parser = argparse.ArgumentParser(description="A script dynamically assigns notebook running on idle GPU.")

        parser.add_argument("-no","--no_output", action='store_true', help="Disable notebook output display in terminal.")
        parser.add_argument("-k","--kernel", type=str, default=None, help="Determine the running kernel.")
        # GPU under these 2 threshold will be considered as idle.
        parser.add_argument("-u","--utilt", type=int, default=5, help="GPU utilization threshold (default: 5%%). The lower the value, the stricter the threshold")
        parser.add_argument("-m","--memt", type=int, default=30, help="GPU memory usage threshold (default: 30%%). The lower the value, the stricter the threshold")
        parser.add_argument("-sn","--stanum", type=int, default=1, help="Start number to name the file (default: 1).")
        parser.add_argument("-nl","--no_log", action='store_true', help="Disable notebook output log file.")
        parser.add_argument("-s","--static", action='store_true', help="Use static mode, which will not check GPU utilization and memory usage, but only check if the GPU is not running any task.")
        parser.add_argument("-t","--to_target", action='store_true', help="Save the output notebook to the directory of notebook instead of the current working directory.")

        try:
            args = parser.parse_args()
        except SystemExit:
            # Use default args for testing without command-line input
            # args = argparse.Namespace(o2t=True, kernel="kernel3",utilt=10,memt=60, stanum=1)
            sys.exit()

        self.kernel=args.kernel
        self.output2terminal=not args.no_output
        self.util_threshold = args.utilt
        self.memory_threshold = args.memt / 100
        self.start_num=args.stanum
        self.log=not args.no_log
        self.static_mode = args.static
        self.save_to_target = args.to_target

    def execute_task(self,notebook, gpu_id, params, task_count,event):
        
        start_time = time.time()

        time.sleep(5)

        notebook_dir = os.path.dirname(os.path.abspath(notebook))

        # if notebook_dir not in sys.path:
        #     sys.path.append(notebook_dir)

        working_dir = notebook_dir if self.save_to_target else self.cwd
        print(f"Working directory: {working_dir}")

        notebook_name = os.path.basename(notebook)
        notebook_name = os.path.splitext(notebook_name)[0]  # Remove the .ipynb extension

        output_dir = os.path.join(working_dir, "results", f"{notebook_name}_output")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir,exist_ok=True)
        
        output_index=get_next_demo_index(output_dir, f"{notebook_name}", ".ipynb", strict=False)
        output_notebook = os.path.join(
            output_dir,
            f"{notebook_name}_task{task_count}_{output_index}.ipynb"
        )


        command = [
            "papermill",
            notebook,
            output_notebook,
        ]

        if self.kernel is not None:
            command.extend(["--kernel", self.kernel])

        for key, value in params.items():
            command.extend(["-p", key, str(value)])
        
        if self.output2terminal is True:
            command.extend(["--log-output"])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        python_path = notebook_dir
        if "PYTHONPATH" in env:
            python_path += os.pathsep + env["PYTHONPATH"]
        env["PYTHONPATH"] = python_path

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            env=env, 
            cwd=working_dir
        )
        
        time.sleep(5)   # In case that threads start together.

        event.set()

        if self.log:
            with open(os.path.join(working_dir, f"{notebook_name}_{task_count}_log.txt"), "a") as logfile:
                for line in process.stdout:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    output = f"[{timestamp}][Task:{task_count}][{notebook} on GPU {gpu_id}] {line.strip()}"
                    print(output)
                    logfile.write(output + "\n")
                    logfile.flush()
            
        process.wait()

        end_time = time.time()
        execution_time = end_time - start_time
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}][Task {task_count}]: {notebook} on GPU {gpu_id} completed in {execution_time:.2f} seconds")
        if gpu_id in self.gpu_active:
            self.gpu_active.remove(gpu_id)


    def get_gpu_status(self):
        """
        Get the usage and memory usage of each GPU.
        Return list: [(gpu_id, utilization, memory_used, memory_total), ...]
        """
        command = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,nounits,noheader"]
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True,)
        lines = result.stdout.strip().split("\n")
        gpu_status = []
        for line in lines:
            parts = line.split(", ")
            gpu_id = int(parts[0])
            utilization = int(parts[1]) 
            memory_used = int(parts[2]) 
            memory_total = int(parts[3])
            gpu_status.append((gpu_id, utilization, memory_used, memory_total))
        return gpu_status
    
    def assign_and_exe(self):
        tasks = self.notebooks_tasks[:]
        threads = []
        gpu_last_assign = {}  # gpu_id: last_assign_timestamp
        self.gpu_active = []
        task_count=self.start_num-1
        # Add task number to avoid overwritten the file with same name,
        # while training the same code but with different args.

        
        while tasks:
            gpu_status = self.get_gpu_status()
            # Sort by GPU utilization and allocate to idle GPUs first
            gpu_status.sort(key=lambda x: (x[1], x[2]))  # Sort by utilization and used video memory
            # Because system and the main process run on GPU 0, so by using sorting, the tasks will be started from GPU 1.

            for gpu_id, utilization, memory_used, memory_total in gpu_status:
                task_start = gpu_id not in self.gpu_active if self.static_mode \
                    else utilization < self.util_threshold and memory_used < memory_total * self.memory_threshold # If GPU condition is below this circumstance, it'll be considered as idle.

                if task_start:  
                    # Check if the GPU was assigned a task recently

                    if gpu_id in self.gpu_active: # Double check
                        continue

                    if not self.static_mode and gpu_id in gpu_last_assign:
                        if time.time() - gpu_last_assign[gpu_id] < 60:
                            continue 
                    else:
                        if len(tasks) ==0:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"[{timestamp}][Task Manager] All tasks are assigned!")
                            break
                        else:
                            notebook, params = tasks.pop(0)
                        event=Event()
                        task_count+=1
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}][Task Manager] Task {task_count}: Assigning {notebook} to GPU {gpu_id} (Utilization: {utilization}%, Memory Used: {memory_used}/{memory_total} MB)")
                        thread = threading.Thread(target=self.execute_task, args=(notebook, gpu_id, params, task_count, event), daemon=True)
                        # daemon=True allows the thread to exit when the main program exits
                        self.gpu_active.append(gpu_id)
                        thread.start()
                        event.wait()
                        threads.append(thread)
                        gpu_last_assign[gpu_id] = time.time()
                        time.sleep(60) # Wait for the assigned task to start, which prevents the order is messed up. 
                        continue
                else:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}][Task Manager] GPU {gpu_id} is busy (Utilization: {utilization}%, Memory Used: {memory_used}/{memory_total} MB). Waiting for idle GPU...")
                    continue
            print(f"[{timestamp}][Task Manager] Currently no GPU is available.")
            time.sleep(10) # Check interval

        for thread in threads:
            thread.join()

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}][Task Manager] All tasks are done!")

if __name__ == "__main__":
    manager=DynamicAssignTaskOnGPU()