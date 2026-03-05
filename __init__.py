"""
OITOOLS - Python interferometry tools for optical interferometry data.

Pure Python implementation of NFFT-based image reconstruction for 
optical/infrared interferometry.

Adapted from the Julia based OITOOLS.jl package by Fabien Baron.
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
from .create_oifits import (
    NoiseConfig,
    project_baseline_to_uv,
    generate_uv_sampling,
    sample_model_observables,
    create_oifits_from_model
)
from .image_to_observables import (
    image_to_cvis_grid,
    make_image_cvis_model,
    sample_image_observables,
    create_oifits_from_image
)
from .plot_oifits import (
    plot_vis_vs_baseline,
    plot_vis2_vs_baseline,
    plot_t3_vs_baseline,
    plot_uv_coverage,
    plot_observables_overview
)

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
    'visibility_ud',
    'NoiseConfig',
    'project_baseline_to_uv',
    'generate_uv_sampling',
    'sample_model_observables',
    'create_oifits_from_model',
    'image_to_cvis_grid',
    'make_image_cvis_model',
    'sample_image_observables',
    'create_oifits_from_image',
    'plot_vis_vs_baseline',
    'plot_vis2_vs_baseline',
    'plot_t3_vs_baseline',
    'plot_uv_coverage',
    'plot_observables_overview'
]

__version__ = '1.0.0'
