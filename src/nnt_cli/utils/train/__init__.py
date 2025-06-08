# AUTO-GENERATED SECTION START - OVERWRITE POSSIBLE
import importlib
LAZY_MODULES = [
    'gen_train',
    'snn_train',
]
def __getattr__(name):
    if name in LAZY_MODULES:
        mod = importlib.import_module(f'.{name}', __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f'module {__name__} has no attribute {name}')

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import gen_train
    from . import snn_train

# AUTO-GENERATED SECTION END
