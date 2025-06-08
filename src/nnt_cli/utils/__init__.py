# AUTO-GENERATED SECTION START - OVERWRITE POSSIBLE
import importlib
LAZY_MODULES = [
    'layer',
    'parallel_run',
    'plot',
    'settin',
    'train',
    'transform',
    'collate_fn',
    'data_analysis',
    'data_trans',
    'distrib',
    'sl',
]
def __getattr__(name):
    if name in LAZY_MODULES:
        mod = importlib.import_module(f'.{name}', __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f'module {__name__} has no attribute {name}')

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import layer
    from . import parallel_run
    from . import plot
    from . import settin
    from . import train
    from . import transform
    from . import collate_fn
    from . import data_analysis
    from . import data_trans
    from . import distrib
    from . import sl

# AUTO-GENERATED SECTION END
