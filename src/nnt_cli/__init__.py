# AUTO-GENERATED SECTION START - DO NOT EDIT
from . import core
from . import custom_templates
from . import custom_utils
from . import templates
from . import utils
from . import cli
# AUTO-GENERATED SECTION END


from pathlib import Path

# CONFIG_DIR = Path.home() / ".nnt_cli"
# CONFIG_DIR.mkdir(parents=True, exist_ok=True)

from .core.funct import CUS_TDIR, CUS_UDIR

custom_tdir = CUS_TDIR
custom_udir = CUS_UDIR

custom_tdir.mkdir(exist_ok=True)
custom_udir.mkdir(exist_ok=True)
