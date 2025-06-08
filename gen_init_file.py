# To manually generate/update __init__.py
import sys
sys.path.append("D:/SoftProject/Python/NeoNetTemp")
from nnt_cli.core.gen_init import generate_package_init
generate_package_init("src/nnt_cli",recursive=True, mode='lazy')
generate_package_init("src/nnt_cli/core",recursive=True, mode='direct',exclude_files=['utils_shell.py'])