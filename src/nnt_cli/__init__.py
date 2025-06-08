# AUTO-GENERATED SECTION START - OVERWRITE POSSIBLE
import importlib
LAZY_MODULES = [
    'core',
    'custom',
    'templates',
    'utils',
    'cli',
    'path',
]
def __getattr__(name):
    if name in LAZY_MODULES:
        mod = importlib.import_module(f'.{name}', __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f'module {__name__} has no attribute {name}')

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import core
    from . import custom
    from . import templates
    from . import utils
    from . import cli
    from . import path

# AUTO-GENERATED SECTION END

__version__ = "0.4.11"

import os

from nnt_cli.path import CUS_DIR, CUS_TDIR, CUS_UDIR


custom_dir=CUS_DIR
custom_tdir = CUS_TDIR
custom_udir = CUS_UDIR

os.makedirs(custom_dir, exist_ok=True)
os.makedirs(custom_tdir, exist_ok=True)
os.makedirs(custom_udir, exist_ok=True)
