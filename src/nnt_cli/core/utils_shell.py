# Load nnt_cli.utils functions separately
from nnt_cli.utils.parallel_run.parallel_run import ParallelNotebookRunner
from nnt_cli.utils.parallel_run.parallel_run_dyn import DynamicAssignTaskOnGPU

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