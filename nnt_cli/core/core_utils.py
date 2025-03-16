from pathlib import Path

def find_files(template_name, search_paths):
    """Search for template files in multiple directories"""
    found_files = []
    for path in search_paths:
        search_path = Path(path) / template_name
        if search_path.exists():
            found_files.append(search_path)
            
        for f in Path(path).rglob(template_name):
            if f.is_file():
                found_files.append(f)
    if not found_files:
        raise FileNotFoundError(f"Can't find {template_name} in directories.")

    return found_files

def select_template_interactive(found_files):
    """Let the user choose when multiple templates of the same name are found"""
    print(f"Found {len(found_files)} templates of the same name:")
    for i, path in enumerate(found_files, 1):
        print(f"{i}. {path}")

    while True:
        try:
            choice = int(input("Please select the template to use (enter the number):"))
            if 1 <= choice <= len(found_files):
                return found_files[choice-1]
            print("Entering the number out of range, please re-enter")
        except ValueError:
            print("Please enter a valid number")