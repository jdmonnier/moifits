"""
OITOOLS - Python interferometry tools for optical interferometry data.

Pure Python implementation of NFFT-based image reconstruction for 
optical/infrared interferometry.
"""

from .readoifits import readoifits
from .oichi2 import (
    setup_nfft,
    image_to_vis,
    image_to_obs,
    chi2_nfft,
    chi2_fg,
    nfft_adjoint,
    mod360
)
from .oioptimize import (
    chi2_sparco_f,
    chi2_sparco_fg,
    optimize_sparco_parameters
)
from .vis_functions import visibility_ud

__all__ = [
    'readoifits',
    'setup_nfft',
    'image_to_vis',
    'image_to_obs',
    'chi2_nfft',
    'chi2_fg',
    'nfft_adjoint',
    'mod360',
    'chi2_sparco_f',
    'chi2_sparco_fg',
    'optimize_sparco_parameters',
    'visibility_ud'
]

__version__ = '1.0.0'
