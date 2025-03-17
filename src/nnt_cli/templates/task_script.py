import subprocess
import threading
import os
import argparse
import time

import sys
import os

current_file = __file__
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from nnt_cli.utils.parallel_run.parallel_run_dyn import DynamicAssignTaskOnGPU

class DynTaskRunner__Project_(DynamicAssignTaskOnGPU):
    def set_task(self):

        self.notebooks_task = [
            # ("file_name.ipynb", {parameters settings in dictionary form}),
            # Examples:
            # ("L4_no_re.ipynb", {"script_mode": True, "train_method" : "one", "num_hiddens" : 512, "num_epochs" : 100, "bit_width" : [8,4,2,1],"match_name":"L4_mixed_bit_width_"}),
            # ("L4_no_re.ipynb", {"script_mode": True, "train_method" : "one", "num_hiddens" : 256, "num_epochs" : 100, "bit_width" : [8,4,2,1],"match_name":"L4_mixed_bit_width_"}),

        ]

if __name__ == "__main__":
    runner = DynTaskRunner__Project_()