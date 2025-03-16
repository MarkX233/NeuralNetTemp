import subprocess
import threading
import os
import argparse


class ParallelNotebookRunner:
    def __init__(self):
        self.get_args()
        self.set_tasks()
        self.run_tasks()

    def get_args(self):
        parser = argparse.ArgumentParser(description="A script assigns notebook running on parallel GPU.")

        parser.add_argument("--o2t", type=bool, default=True, help="Enable or disable notebook output display in terminal.")
        parser.add_argument("--kernel", type=str, default="kernel3", help="Determine the running kernel.")

        try:
            args = parser.parse_args()
        except SystemExit:
            # Use default args for testing without command-line input
            args = argparse.Namespace(o2t=True, kernel="kernel3")

        self.kernel = args.kernel
        self.output2terminal = args.o2t

    def set_tasks(self):
        """
        notebooks_and_params (list): A list of tuples where each tuple contains:
            - notebook (str): The name of the notebook file to run.
            - gpu_id (int): The GPU ID to use for running the notebook.
            - parameters (dict): A dictionary of parameters to pass to the notebook.
                                 The parameters need to be first set in the certain cell with `parameters` tag in the notebook.
        """
        self.notebooks_and_params = [
            ("L2.ipynb", 0, {"para_mode": True, "num_epochs": 3}),
            ("L3.ipynb", 1, {"para_mode": True, "num_epochs": 3}),
            ("L4.ipynb", 2, {"para_mode": True, "num_epochs": 3}),
        ]

    def stream_output_with_prefix(self, process, prefix, logfile):
        for line in iter(process.stdout.readline, ""):
            output = f"{prefix} {line.strip()}"
            print(output)
            logfile.write(output + "\n")
            logfile.flush()

    def run_tasks(self):
        processes = []
        task_count = 0

        for notebook, gpu_id, parameters in self.notebooks_and_params:
            task_count += 1
            output_notebook = notebook.replace(".ipynb", f"_output_{task_count}.ipynb")
            command = [
                "papermill",
                notebook,
                output_notebook,
                "--kernel", self.kernel,
            ]
            for key, value in parameters.items():
                command.extend(["-p", key, str(value)])

            if self.output2terminal:
                command.append("--log-output")

            logfile = open(f"{notebook}_log.txt", "w")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,  # Use `text=True` for Python 3.7+ instead of `universal_newlines`
                env=env,
            )
            processes.append((process, logfile))

            thread = threading.Thread(
                target=self.stream_output_with_prefix,
                args=(process, f"[{notebook}]", logfile),
            )
            thread.daemon = True
            thread.start()

        for process, logfile in processes:
            process.wait()
            logfile.close()

        print("All processes are done!")


if __name__ == "__main__":
    runner = ParallelNotebookRunner()
