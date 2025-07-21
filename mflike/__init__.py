from importlib.metadata import PackageNotFoundError, version

from .foreground import (
    BandpowerForeground,
    EEForeground,
    Foreground,
    TEEEForeground,
    TEForeground,
    TTEEForeground,
    TTForeground,
    TTTEForeground,
)
from .mflike import EE, TE, TEEE, TT, TTEE, TTTE, TTTEEE

try:
    __version__ = version("mflike")
except PackageNotFoundError:
    # package is not installed
    pass
