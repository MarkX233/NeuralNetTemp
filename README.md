# NeuralNetTemp

*This is a side project of my Studien Arbeit, so some code may not be fully tested.*

## Introduction

NeuralNetTemp, nnt stands for Neural Network Template, is a template manage system along with the templates for creating neural networks in Python. The goal of NeuralNetTemp is to provide simple and easy-to-use templates with useful utils to let one only need to focus on creating neural network model, while still allowing for a high degree of customization and flexibility.
It is built on top of the popular [PyTorch](https://pytorch.org/) library, which provides a wide range of tools for building and training neural networks.

## Quick Start

### 1. Installation

As the package is not uploaded to PyPI, you need to download the wheel file from the release page and install it manually.

```bash
pip install nnt_cli-0.3.0-py3-none-any.whl
```

After the installation, you can use the command below to see whether is installed properly.

```bash
nnt --help
```

### 2. Start a project

You can start a new project by using the following command.

```bash
nnt cp my_project
```

This command will create a new project folder named `my_project` in the current directory. The project folder contains the following structure.

```
my_project
├── pro_temp.py
├── model.ipynb
├── task_script.py
```

In the `my_project` folder, `pro_temp.py` is the template file for the project, here you can defined parts can be shared between different tasks, such as dataset and dataloader etc.. `pro_temp.py` inherits from `gen_temp.py` which is the general template file with the basic settings that can be shared between projects. In `gen_temp.py`, the basic training loop, plot function, save function and some other useful functions are defined. 

`model.ipynb` is the Jupyter notebook file with specific settings for a model. `model.ipynb` inherits from pro_temp.py and gen_temp.py, and it is the place where you can define the model and the training process. You can use multiple `model.ipynb` files to define different models for the same project.

`task_script.py` is the script file for the task. By defining the task in the `task_script.py`, you can pass parameters to `model.ipynb`. Therefore, if models are only different in some hyperparameters, you can use the same `model.ipynb` file and pass the hyperparameters from the `task_script.py`.

### 3. Run the project

You can directly use:

```bash
python task_script.py
```

to run the task. But most importantly, you must set the kernel with `--kernel` before running the task.

```bash
python task_script.py --kernel your_kernel_name
```

Use `python task_script.py --help` to see the help message of task script.

After calling the task script, the `model.ipynb` will be executed with the parameters passed from the task script. Task script will automatically allocate the tasks on the available GPUS, and the results will be saved in the `result` folder in the project folder.

If you have multiple projects to run, you can use a total project script to run all the tasks in all the projects. To do this, go to the parent folder of all the projects and use the following command:

```bash
nnt cf run_all_projects.sh
```

to copy the `run_all_projects.sh` to the parent folder. You need to modify the task project list in the `run_all_projects.sh` file before running. Then you can simply call the `run_all_projects.sh`, and this scrip will run all the `task_script.py` in all the project folders.

### Code management between works

Sometimes, you may want to reuse the code you wrote in the previous works. You can use the following command to add your custom code to a global package.

```bash
# To save in the `nnt_cli.custom.custom_template`
nnt sf your_custom_template.py
```

or

```bash
# To save in the `nnt_cli.custom.custom_utils`
nnt sf your_custom_code.py -u
```

After saving the custom code, use:

```bash
nnt lf
```

to list all available custom code and internal templates. You can use the following command to copy the custom code to the project folder.

```bash
nnt cp custom_code_name_with_extension project_folder
```

The program will update `__init__.py`, every time you change the custom code with `nnt` commands. So if you don't need to modify the custom code, you can directly use `import nnt_cli.custom.custom_template/utils.your_custom_template` to import the custom code into the project code.

Import and export the whole custom code is also supported. You can use the following command to export the custom code to a file.

```bash
# Export the custom code
nnt ec
# Import the custom code
nnt ic custom_code_folder
```

### Using Git to sync the custom code

This is an optional step, but it is recommended to use git to sync the custom code in case you lost them.

The implementation of the git in `nnt_cli` is simply a proxy of git. Which means, using `nnt git` command is basically the same as the git command, but it's target is the custom code folder.


First, you need to initialize the nnt custom folder.

```bash
nnt git init
```

If you didn't use git before, you need to set up the git account.

```bash
nnt git config user.email "your.email@example.com"
nnt git config user.name "Your Name"
```

Then, you need to create a git repository to store your custom code and link that remote repository to your custom folder. 

```bash
nnt git remote add origin git@github.com:yourname/repo-name.git
```

Before you use git to manage your custom code, you need to creat a gitingore file to ignore the pyc and log files. I suggest you to use the following command.

```bash
nnt sync --first
```
It will create a `.gitignore` file in the custom folder and ignore the pyc and log files. You can also use `nnt sync` to sync your local files to the remote repository for a quick sync. You can also use the git command to manage your custom code. It's the same as the git command, but use `nnt` before `git`, so the custom code you saved will be operated.

