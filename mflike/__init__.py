from importlib.metadata import PackageNotFoundError, version

from .foreground import BandpowerForeground, Foreground, EEForeground, TEForeground, TTForeground
from .mflike import TT, TE, EE, TTTEEE

try:
    __version__ = version("mflike")
except PackageNotFoundError:
    # package is not installed
    pass
