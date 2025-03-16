import argparse

from core.funct import *

def main():

    # import sys
    # sys.path.append("D:/SoftProject/Python/NeoNetTemp")

    parser = argparse.ArgumentParser(description="A quick template generator.")

    subparsers = parser.add_subparsers()

    # save-files, default is template
    save_parser = subparsers.add_parser("save-file", 
                                        aliases=["sf"],
                                        help="Save self defined Template to the global template directory.")
    save_parser.add_argument("file_path", help="To save file path")
    save_parser.add_argument("-f", "--force", action="store_true",
        help="Overwrite existed template")
    save_parser.add_argument("-u", "--utils", action="store_true",
        help="Save into custom utils directory.")
    save_parser.set_defaults(func=save_files)

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
    copy_parser.add_argument("target_dir", default=".", help="target directory")
    copy_parser.set_defaults(func=copy_template)

    # create-project
    create_parser = subparsers.add_parser(
        "create-project",
        aliases=["cp"],
        help="Create a new project from template"
    )
    create_parser.add_argument("project_name", help="Project name")
    create_parser.add_argument("-t","--template_name", help="Template name")
    create_parser.set_defaults(func=create_project)

    
    
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()