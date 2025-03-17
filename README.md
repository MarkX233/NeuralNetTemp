# NeuralNetTemp

*This is a side project of my Studien Arbeit, some code may not be fully tested.*

## Introduction

NeuralNetTemp, nnt stands for Neural Network Template, is a template manage system along with the templates for creating neural networks in Python. The goal of NeuralNetTemp is to provide simple and easy-to-use templates with useful utils to let one only need to focus on creating neural network model, while still allowing for a high degree of customization and flexibility.
It is built on top of the popular [PyTorch](https://pytorch.org/) library, which provides a wide range of tools for building and training neural networks.

## Quick Start

### 1. Installation

```bash
pip install nnt_cli
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
It will create a `.gitignore` file in the custom folder and ignore the pyc and log files. You can also use `nnt sync` to sync your local files to the remote repository for a quick sync.

Now you can use the git command to manage your custom code. It's the same as the git command, but use `nnt` before `git`, so the custom code you saved will be operated.

