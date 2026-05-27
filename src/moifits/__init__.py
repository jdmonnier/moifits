"""
Michigan OI-FITS tools.

The package keeps imports lazy so lightweight readers and plotting commands do
not need optimizer or NFFT dependencies until those functions are requested.
"""

from importlib import import_module

__version__ = "1.0.0"

_EXPORTS = {
    "readoifits": (".readoifits", "readoifits"),
    "NFFTPlan": (".oichi2", "NFFTPlan"),
    "setup_nfft": (".oichi2", "setup_nfft"),
    "image_to_vis": (".oichi2", "image_to_vis"),
    "image_to_obs": (".oichi2", "image_to_obs"),
    "chi2_nfft": (".oichi2", "chi2_nfft"),
    "chi2_fg": (".oichi2", "chi2_fg"),
    "nfft_adjoint": (".oichi2", "nfft_adjoint"),
    "mod360": (".oichi2", "mod360"),
    "chi2_sparco_f": (".oioptimize", "chi2_sparco_f"),
    "chi2_sparco_fg": (".oioptimize", "chi2_sparco_fg"),
    "optimize_sparco_parameters": (".oioptimize", "optimize_sparco_parameters"),
    "visibility_ud": (".vis_functions", "visibility_ud"),
    "NoiseConfig": (".writeoifits", "NoiseConfig"),
    "project_baseline_to_uv": (".writeoifits", "project_baseline_to_uv"),
    "generate_uv_sampling": (".writeoifits", "generate_uv_sampling"),
    "sample_model_observables": (".writeoifits", "sample_model_observables"),
    "create_oifits_from_model": (".writeoifits", "create_oifits_from_model"),
    "image_to_cvis_grid": (".image_to_observables", "image_to_cvis_grid"),
    "make_image_cvis_model": (".image_to_observables", "make_image_cvis_model"),
    "sample_image_observables": (".image_to_observables", "sample_image_observables"),
    "create_oifits_from_image": (".image_to_observables", "create_oifits_from_image"),
    "plot_vis_vs_baseline": (".plot_oifits", "plot_vis_vs_baseline"),
    "plot_vis2_vs_baseline": (".plot_oifits", "plot_vis2_vs_baseline"),
    "plot_t3_vs_baseline": (".plot_oifits", "plot_t3_vs_baseline"),
    "plot_uv_coverage": (".plot_oifits", "plot_uv_coverage"),
    "plot_observables_overview": (".plot_oifits", "plot_observables_overview"),
    "ComponentSpec": (".models", "ComponentSpec"),
    "HaloSpec": (".models", "HaloSpec"),
    "ModelSpec": (".models", "ModelSpec"),
    "ParameterSpec": (".models", "ParameterSpec"),
    "SparcoSpec": (".models", "SparcoSpec"),
    "StarSpec": (".models", "StarSpec"),
    "free_parameter_names": (".models", "free_parameter_names"),
    "free_parameters": (".models", "free_parameters"),
    "load_model_spec": (".models", "load_model_spec"),
    "make_coordinate_grid": (".models", "make_coordinate_grid"),
    "parameter_bounds": (".models", "parameter_bounds"),
    "parameter_vector": (".models", "parameter_vector"),
    "render_gaussian": (".models", "render_gaussian"),
    "render_model": (".models", "render_model"),
    "render_ring": (".models", "render_ring"),
    "single_component_model": (".models", "single_component_model"),
    "update_model_parameters": (".models", "update_model_parameters"),
    "DISK_SPARCO_PARAM_NAMES": (".fitting", "DISK_SPARCO_PARAM_NAMES"),
    "log_posterior_model_vector": (".fitting", "log_posterior_model_vector"),
    "log_prior_model_vector": (".fitting", "log_prior_model_vector"),
    "render_disk": (".fitting", "render_disk"),
    "run_emcee_model_fit": (".fitting", "run_emcee_model_fit"),
    "chi2_sparco_model": (".sparco", "chi2_sparco_model"),
    "model_complex_visibility": (".sparco", "model_complex_visibility"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
