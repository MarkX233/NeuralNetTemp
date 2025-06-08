import pathlib
from typing import Union, Optional, List
import re

def generate_package_init(
    pkg_path: Union[str, pathlib.Path],
    recursive: bool = False,
    parent_pkg: Optional[str] = None,
    exclude_dirs: List[str] = None,
    exclude_files: List[str] = None,
    echo: bool = True,
    mode: str = 'direct',
    type_checking: bool = True
) -> None:
    """
    Automatically generate the __init__.py file of the package, explicitly import all subpackages and modules (IDE-friendly)

    Args:
        pkg_path: Path to the package directory (string or pathlib.Path object)
        recursive: Whether to recursively process subpackages of subpackages (default False)
        parent_pkg: Full path to the parent package (used when recursive internally, without manual specification)
        exclude_dirs: Excluded directory names (such as ["tests", "docs"])
        exclude_files: Excluded file name (such as ["_private.py"])
        echo: Do not print details.
        mode: `direct`: from . import modules/packages \\
            `all`: __all__ = [modules/packages] \\
            `lazy`: lazy import using importlib and __getattr__
    Example:
        generate_package_init("your_package", recursive=True)
    """
    pkg_dir = pathlib.Path(pkg_path)
    current_pkg = parent_pkg or pkg_dir.name
    exclude_dirs = exclude_dirs or []
    exclude_files = exclude_files or []

    if not pkg_dir.exists():
        raise FileNotFoundError(f"Path '{pkg_dir}' not exist.")
    if not pkg_dir.is_dir():
        raise NotADirectoryError(f"'{pkg_dir}' is not a path.")
    
    subpackages = []
    modules = []

    for child in pkg_dir.iterdir():
        if child.name.startswith(".") or any(c in child.name for c in ["__", "#"]):
            continue
        if child.name in exclude_dirs + exclude_files:
            continue

        if child.is_dir():
            subpackages.append(child.name)
        elif child.suffix == ".py" and child.stem != "__init__":
            modules.append(child.stem)
    
    init_content = []
    if mode == 'direct':
        if subpackages:
            init_content.extend(f"from . import {name}" for name in sorted(subpackages))
        if modules:
            init_content.extend(f"from . import {name}" for name in sorted(modules))
        auto_content = "\n".join(init_content)
    elif mode == 'all':
        init_content.append("__all__ = [\n")
        if subpackages:
            init_content.extend([f"    '{name}',\n" for name in sorted(subpackages)])
        if modules:
            init_content.extend([f"    '{name}',\n" for name in sorted(modules)])
        init_content.append("]\n")
        auto_content = "".join(init_content)
    elif mode == 'lazy':
        # import importlib
        init_content.append("import importlib\n")
        init_content.append("LAZY_MODULES = [\n")
        if subpackages:
            init_content.extend([f"    '{name}',\n" for name in sorted(subpackages)])
        if modules:
            init_content.extend([f"    '{name}',\n" for name in sorted(modules)])
        init_content.append("]\n")
        init_content.append(
            "def __getattr__(name):\n"
            "    if name in LAZY_MODULES:\n"
            "        mod = importlib.import_module(f'.{name}', __name__)\n"
            "        globals()[name] = mod\n"
            "        return mod\n"
            "    raise AttributeError(f'module {__name__} has no attribute {name}')\n"
        )

        if type_checking:
            init_content.append("\nfrom typing import TYPE_CHECKING\n\n")
            init_content.append("if TYPE_CHECKING:\n")
            if subpackages:
                init_content.extend(f"    from . import {name}\n" for name in sorted(subpackages))
            if modules:
                init_content.extend(f"    from . import {name}\n" for name in sorted(modules))
        auto_content = "".join(init_content)


    MARKER_START = "# AUTO-GENERATED SECTION START - OVERWRITE POSSIBLE\n"
    MARKER_END = "# AUTO-GENERATED SECTION END\n"
    marked_content = f"{MARKER_START}{auto_content}\n{MARKER_END}"

    init_file = pkg_dir / "__init__.py"
    original_content = ""
    if init_file.exists():
        with init_file.open("r", encoding="utf-8") as f:
            original_content = f.read()

    pattern = re.compile(
        rf"{re.escape(MARKER_START)}.*?{re.escape(MARKER_END)}",
        flags=re.DOTALL
    )

    if pattern.search(original_content):
        new_content = pattern.sub(marked_content, original_content, count=1)
    else:
        new_content = f"{marked_content}\n\n{original_content}"

    with init_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write(new_content.strip() + "\n")

    if echo is True:
        print(f"Update {init_file}: {len(subpackages)} subpackages, {len(modules)} modules")

    if recursive:
        for sub in subpackages:
            sub_path = pkg_dir / sub
            generate_package_init(
                sub_path,
                recursive=True,
                parent_pkg=f"{current_pkg}.{sub}",
                exclude_dirs=exclude_dirs,
                exclude_files=exclude_files,
                mode=mode
            )