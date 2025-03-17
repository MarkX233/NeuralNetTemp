import subprocess
import threading
import os
import argparse
import time

parser = argparse.ArgumentParser(description="A script dynamically assigns notebook running on idle GPU.")

parser.add_argument("--o2t", type=bool, default=True, help="Enable or disable notebook output display in terminal.")
parser.add_argument("--kernel", type=str, default="kernel3", help="Determine the running kernel.")

notebooks_task = [
    ("L2.ipynb", {"para_mode": True, "num_epochs" : 5}),
    ("L3.ipynb", {"para_mode": True, "num_epochs" : 5}),
    ("L4.ipynb", {"para_mode": True, "num_epochs" : 5}),
]

"""
notebooks_and_params (list): A list of tuples where each tuple contains:
    - notebook (str): The name of the notebook file to run.
    - parameters (dict): A dictionary of parameters to pass to the notebook.
                         The parameters need to be first set in the certain cell with `parameters` tag in the notebook.
"""

args = parser.parse_args()

kernel=args.kernel

output2terminal=args.o2t

def execute_task(notebook, gpu_id, params, task_count):
    output_notebook = notebook.replace(".ipynb", f"_output_{task_count}.ipynb")  
    command = [
        "papermill",
        notebook,
        output_notebook,
        "--kernel", kernel,
    ]
    for key, value in params.items():
        command.extend(["-p", key, str(value)])
    
    if output2terminal is True:
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
    
    for line in process.stdout:
        output=f"[{notebook} on GPU {gpu_id}] {line.strip()}"
        print(output)
        with open(f"{notebook}_{task_count}_log.txt", "a") as logfile:
            logfile.write(output + "\n")
            logfile.flush()
        


    process.wait()


def get_gpu_status():
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

tasks = notebooks_task[:]
threads = []
last_gpu=None
task_count=0
# Add task number to avoid overwritten the file with same name,
# if you want train the same code but with different args.
while tasks:
    gpu_status = get_gpu_status()
    # Sort by GPU utilization and allocate to idle GPUs first
    gpu_status.sort(key=lambda x: (x[1], x[2]))  # Sort by utilization and used video memory

    for gpu_id, utilization, memory_used, memory_total in gpu_status:
        if utilization < 10 and memory_used < 1000:  # If GPU condition is below this circumstance, it'll be considered as idle.
            if gpu_id == last_gpu:  # To avoid that the recent assigned task haven't started.
                time.sleep(20)
                last_gpu=None
                break
            else:
                if len(tasks) ==0:
                    print("All tasks are assigned!")
                    break
                else:
                    notebook, params = tasks.pop(0)
                task_count+=1
                print(f"Assigning {notebook} to GPU {gpu_id} (Utilization: {utilization}%, Memory Used: {memory_used}/{memory_total} MB)")
                thread = threading.Thread(target=execute_task, args=(notebook, gpu_id, params, task_count))
                thread.start()
                threads.append(thread)
                last_gpu=gpu_id
                continue
    time.sleep(10) # Check interval

for thread in threads:
    thread.join()

print("All tasks are done!")