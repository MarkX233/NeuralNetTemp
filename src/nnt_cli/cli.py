import argparse

from core.funct import save_files, del_files, list_files, copy_template, \
    create_project, export_custom, import_custom, git_proxy, sync_command

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

    # copy-template
    copy_parser = subparsers.add_parser(
        "copy-template",
        aliases=["ct"],
        help="Copy template to target directory"
    )
    copy_parser.add_argument("template_name", help="template name")
    copy_parser.add_argument("target_dir", nargs="?", default=".", help="target directory")
    copy_parser.set_defaults(func=copy_template)

    # create-project
    create_parser = subparsers.add_parser(
        "create-project",
        aliases=["cp"],
        help="Create a new project from template"
    )
    create_parser.add_argument("project_name", help="Project name")
    create_parser.add_argument("target_dir", nargs="?", default=".", help="Target directory")
    create_parser.add_argument("-t","--template_name", default=None, help="Template name")
    create_parser.set_defaults(func=create_project)

    export_parser = subparsers.add_parser(
        "export-custom",
        aliases=["ec"],
        help="Export custom code"
    )
    export_parser.add_argument("export_dir", nargs="?", default=".", help="Target export directory")
    export_parser.set_defaults(func=export_custom)

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
    sync_parser.set_defaults(func=sync_command)



    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
