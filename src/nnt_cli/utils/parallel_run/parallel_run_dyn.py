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
        self.get_args()
        self.set_task()
        get_notebook_name("./",raise_error=False,sub_dir=True)
        self.assign_and_exe()

    def set_task(self):
        """
        notebooks_and_params (list): A list of tuples where each tuple contains:
            - notebook (str): The name of the notebook file to run.
            - parameters (dict): A dictionary of parameters to pass to the notebook.
                                The parameters need to be first set in the certain cell with `parameters` tag in the notebook.
        """
        self.notebooks_tasks = [
            ("L2.ipynb", {"para_mode": True, "num_epochs" : 5}),
            ("L3.ipynb", {"para_mode": True, "num_epochs" : 5}),
            ("L4.ipynb", {"para_mode": True, "num_epochs" : 5}),
        ]
    
    def get_args(self):
        parser = argparse.ArgumentParser(description="A script dynamically assigns notebook running on idle GPU.")

        parser.add_argument("--o2t", type=bool, default=True, help="Enable or disable notebook output display in terminal.")
        parser.add_argument("--kernel", type=str, default=None, help="Determine the running kernel.")
        # GPU under these 2 threshold will be considered as idle.
        parser.add_argument("--utilt", type=int, default=5, help="GPU utilization threshold (default: 10%%).")
        parser.add_argument("--memt", type=int, default=30, help="GPU memory usage threshold (default: 60%%).")
        parser.add_argument("--stanum", type=int, default=1, help="Start number to name the file (default: 1).")
        parser.add_argument("--log", type=bool, default=True, help="Enable or disable notebook output log file.")

        try:
            args = parser.parse_args()
        except SystemExit:
            # Use default args for testing without command-line input
            # args = argparse.Namespace(o2t=True, kernel="kernel3",utilt=10,memt=60, stanum=1)
            sys.exit()

        self.kernel=args.kernel
        self.output2terminal=args.o2t
        self.util_threshold = args.utilt
        self.memory_threshold = args.memt / 100
        self.start_num=args.stanum
        self.log=args.log

    def execute_task(self,notebook, gpu_id, params, task_count,event):
        start_time = time.time()

        time.sleep(5)

        notebook_name = notebook.split(".")[0]

        output_dir = f"results/{notebook_name}_output"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir,exist_ok=True)
        
        output_index=get_next_demo_index(output_dir, f"{notebook_name}", ".ipynb", strict=False)
        output_notebook = f"{output_dir}/{notebook_name}_task{task_count}_{output_index}.ipynb"

        # output_notebook = notebook.replace(".ipynb", f"_output_{task_count}.ipynb")  
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

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            env=env, 
        )
        
        time.sleep(5)   # In case that threads start together.

        event.set()

        if self.log:
            with open(f"{notebook}_{task_count}_log.txt", "a") as logfile:
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
        last_gpu=None
        task_count=self.start_num-1
        # Add task number to avoid overwritten the file with same name,
        # while training the same code but with different args.

        while tasks:
            gpu_status = self.get_gpu_status()
            # Sort by GPU utilization and allocate to idle GPUs first
            gpu_status.sort(key=lambda x: (x[1], x[2]))  # Sort by utilization and used video memory

            for gpu_id, utilization, memory_used, memory_total in gpu_status:
                if utilization < self.util_threshold and memory_used < memory_total * self.memory_threshold:  # If GPU condition is below this circumstance, it'll be considered as idle.
                    if gpu_id == last_gpu:  # To avoid that the recent assigned task haven't started.
                        same_gpu_time=time.time()
                        if same_gpu_time-start_time<60:
                            time.sleep(60)
                            # last_gpu=None
                            break
                        else:
                            # last_gpu=None
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
                        thread = threading.Thread(target=self.execute_task, args=(notebook, gpu_id, params, task_count, event))
                        thread.start()
                        event.wait()
                        threads.append(thread)
                        if last_gpu is None:
                            time.sleep(90)  # Initial time for papermill to start kernel at first start.
                        last_gpu=gpu_id
                        start_time = time.time()
                        time.sleep(45) # Wait for the assigned task to start, which prevents the order is messed up. 
                        continue
                else:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}][Task Manager] Currently no GPU is available.")
            time.sleep(10) # Check interval

        for thread in threads:
            thread.join()

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}][Task Manager] All tasks are done!")

if __name__ == "__main__":
    manager=DynamicAssignTaskOnGPU()