from importlib.metadata import PackageNotFoundError, version

from .foreground import BandpowerForeground, Foreground
from .mflike import MFLike_TT, MFLike_TE, MFLike_EE, MFLike_TTTEEE

try:
    __version__ = version("mflike")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = [
    # mflike
    "MFLike_TT",
    "MFLike_TE",
    "MFLike_EE",
    "MFLike_TTTEEE",
    # foreground
    "Foreground",
    "BandpowerForeground",
]
