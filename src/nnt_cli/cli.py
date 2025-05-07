import argparse


from nnt_cli.core.funct import save_files, del_files, list_files, copy_files, \
    create_project, export_custom, import_custom, git_proxy, sync_command, opt_command

def main():

    # import sys
    # sys.path.append("D:/SoftProject/Python/NeoNetTemp")

    parser = argparse.ArgumentParser(description="A quick template generator.")

    subparsers = parser.add_subparsers()

    # save-files, default is template
    save_parser = subparsers.add_parser("save-file", 
                                        aliases=["sf"],
                                        help="Save self defined Template or utils file to the global template directory.")
    save_parser.add_argument("file_path", help="To be saved file path")
    save_parser.add_argument("-f", "--force", action="store_true",
        help="Overwrite existed template")
    save_parser.add_argument("-u", "--utils", action="store_true",
        help="Save into custom utils directory.")
    save_parser.set_defaults(func=save_files)

    # delete-files
    del_parser = subparsers.add_parser("del-file", 
                                        aliases=["df","rf"],
                                        help="Delete self defined Template or utils file in the global template directory.")
    del_parser.add_argument("file_name", help="To be deleted file name")
    del_parser.set_defaults(func=del_files)

    # list-files
    list_parser = subparsers.add_parser(
        "list-files",
        aliases=["lf"],
        help="List all available templates (internal + custom)"
    )
    list_parser.add_argument("-p", "--path", action="store_true",
        help="List the template paths.")
    list_parser.set_defaults(func=list_files)

    # copy-files
    copy_parser = subparsers.add_parser(
        "copy-files",
        aliases=["cf"],
        help="Copy file to target directory"
    )
    copy_parser.add_argument("file_name", help="File name you want to copy")
    copy_parser.add_argument("target_dir", nargs="?", default=".", help="Target directory")
    copy_parser.set_defaults(func=copy_files)

    # create-project
    create_parser = subparsers.add_parser(
        "create-project",
        aliases=["cp"],
        help="Create a new project from template"
    )
    create_parser.add_argument("project_name", help="Project name")
    create_parser.add_argument("target_dir", nargs="?", default=".", help="Target directory")
    create_parser.add_argument("-t","--template_name", default=None, help="Indicate specific template name you want to start.")
    create_parser.set_defaults(func=create_project)
    
    # export-custom-code 
    export_parser = subparsers.add_parser(
        "export-custom",
        aliases=["ec"],
        help="Export custom code"
    )
    export_parser.add_argument("export_dir", nargs="?", default=".", help="Target export directory")
    export_parser.set_defaults(func=export_custom)

    # import-custom-code
    import_parser = subparsers.add_parser(
        "import-custom",
        aliases=["ic"],
        help="Import custom code"
    )
    import_parser.add_argument("import_dir", help="Custom import directory")
    import_parser.set_defaults(func=import_custom)
    
    git_parser = subparsers.add_parser('git', help='Proxy Git commands to custom code directory')
    git_parser.add_argument('git_args', nargs=argparse.REMAINDER, 
                          help='Any Git command parameters')
    git_parser.set_defaults(func=git_proxy)

    sync_parser = subparsers.add_parser('sync', help='One-step synchronization of custom code')
    sync_parser.add_argument('-b','--branch', default='main', help='Target branch')
    # sync_parser.add_argument('-nc','--not_commit', action='store_true', 
    #                        help='Not to automatically submit uncommitted changes')
    sync_parser.add_argument('--init', action='store_true', 
                           help='Init-setup of the remote git repo')
    sync_parser.add_argument('--no_pull', action='store_true', 
                           help='Do not pull form remote')
    sync_parser.add_argument('--no_push', action='store_true', 
                           help='Do not push to remote')
    sync_parser.set_defaults(func=sync_command)

    opt_parser = subparsers.add_parser('opt', help='Automatically optimize the hyperparameters using optuna')
    opt_parser.add_argument('target', help='Target code to optimize, for now only support `.ipynb` file')
    opt_parser.add_argument('-t', '--trial', type=int, default=-1, help='Number of trials, `-1` means use the setting in the code.')
    opt_parser.add_argument('-r', '--thread', type=int, default=-1, help='Number of threads, no more than the number of GPUs, `-1` means use all available GPUs.')
    opt_parser.add_argument('-k', "--kernel", type=str, default=None, help="Determine the running kernel.")
    opt_parser.add_argument('-no', "--no_output", action="store_false", help="Disable output info to terminal.")
    opt_parser.add_argument('-nl', "--no_log", action="store_false", help="Disable output info to log file.")
    opt_parser.add_argument('-s', "--static", type=str, default=None, help="Indicate static GPU to perform the optimization. e.g. [0,1,2] means use GPU 0,1,2 to perform the optimization.")
    opt_parser.set_defaults(func=opt_command)
    
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
