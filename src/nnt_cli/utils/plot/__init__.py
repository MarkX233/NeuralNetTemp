# AUTO-GENERATED SECTION START - OVERWRITE POSSIBLE
import importlib
LAZY_MODULES = [
    'acc_plot',
    'anal_plot',
    'snn_plot',
]
def __getattr__(name):
    if name in LAZY_MODULES:
        mod = importlib.import_module(f'.{name}', __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f'module {__name__} has no attribute {name}')

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import acc_plot
    from . import anal_plot
    from . import snn_plot

# AUTO-GENERATED SECTION END
