# AUTO-GENERATED SECTION START - OVERWRITE POSSIBLE
import importlib
LAZY_MODULES = [
    'QSNN_Template',
    'gen_temp',
    'pro_temp',
    'task_script',
]
def __getattr__(name):
    if name in LAZY_MODULES:
        mod = importlib.import_module(f'.{name}', __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f'module {__name__} has no attribute {name}')

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import QSNN_Template
    from . import gen_temp
    from . import pro_temp
    from . import task_script

# AUTO-GENERATED SECTION END
