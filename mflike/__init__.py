from importlib.metadata import PackageNotFoundError, version

from .foreground import BandpowerForeground, Foreground, EEForeground, TEForeground, TTForeground, TTEEForeground, TTTEForeground, TEEEForeground
from .mflike import TT, TE, EE, TTEE, TTTE, TEEE, TTTEEE

try:
    __version__ = version("mflike")
except PackageNotFoundError:
    # package is not installed
    pass
