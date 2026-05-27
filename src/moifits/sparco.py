"""Model-level SPARCO forward calculations."""

from __future__ import annotations

import numpy as np

from .models import ModelSpec, render_component


def power_law_flux(flux_0: float, spectral_index: float, wavelengths_m: np.ndarray, lambda_0: float) -> np.ndarray:
    """Scale a reference flux with a power law in wavelength."""
    return float(flux_0) * (np.asarray(wavelengths_m, dtype=float) / float(lambda_0)) ** float(spectral_index)


def normalized_component_image(component, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Render one environmental component as unit total flux morphology."""
    image = render_component(component, x_grid, y_grid, include_flux=False)
    total = np.sum(image)
    if total <= 0:
        raise ValueError(f"component {component.name!r} rendered with non-positive total flux")
    return image / total


def model_complex_visibility(
    model: ModelSpec,
    ftplan,
    data,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute model complex visibility with model-level SPARCO scaling.

    The model combines one model-level stellar contribution, an optional
    model-level halo contribution, and all environmental components:

        V = sum(F_i(lambda) V_i) / sum(F_i(lambda))

    The halo visibility defaults to 1.0, i.e. unresolved. Set it to 0.0 in the
    JSON spec to represent fully resolved incoherent flux.
    """
    wavelengths_m = np.asarray(data.uv_lam, dtype=float)
    lambda_0 = model.sparco.lambda_0.value
    numerator = np.zeros(wavelengths_m.shape, dtype=np.complex128)
    denominator = np.zeros(wavelengths_m.shape, dtype=float)

    from .oichi2 import image_to_vis
    from .vis_functions import visibility_ud

    star = model.sparco.star
    star_flux = power_law_flux(star.flux.value, star.spectral_index.value, wavelengths_m, lambda_0)
    numerator += star_flux * visibility_ud([star.diameter.value], data.uv)
    denominator += star_flux

    halo = model.sparco.halo
    if halo is not None:
        halo_flux = power_law_flux(halo.flux.value, halo.spectral_index.value, wavelengths_m, lambda_0)
        numerator += halo_flux * complex(halo.visibility.value)
        denominator += halo_flux

    for component in model.components:
        image = normalized_component_image(component, x_grid, y_grid)
        cvis = image_to_vis(image, ftplan[0])
        env_flux = power_law_flux(
            component.flux.value,
            component.spectral_index.value,
            wavelengths_m,
            lambda_0,
        )
        numerator += env_flux * cvis
        denominator += env_flux

    if np.any(denominator == 0):
        raise ValueError("SPARCO denominator is zero for at least one wavelength")
    return numerator / denominator


def model_observables_from_cvis(cvis_model: np.ndarray, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert full complex visibility samples to VIS2, T3AMP, and T3PHI."""
    from .oichi2 import vis_to_t3, vis_to_v2

    v2_model = vis_to_v2(cvis_model, data.indx_v2)
    _, t3amp_model, t3phi_model = vis_to_t3(
        cvis_model,
        data.indx_t3_1,
        data.indx_t3_2,
        data.indx_t3_3,
    )
    return v2_model, t3amp_model, t3phi_model


def chi2_sparco_model(
    model: ModelSpec,
    ftplan,
    data,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    weights=(1.0, 1.0, 1.0),
    verbose: bool = False,
) -> float:
    """Compute chi-squared for a multi-component model-level SPARCO spec."""
    from .oichi2 import mod360

    cvis_model = model_complex_visibility(model, ftplan, data, x_grid, y_grid)
    v2_model, t3amp_model, t3phi_model = model_observables_from_cvis(cvis_model, data)

    chi2_v2 = chi2_t3amp = chi2_t3phi = 0.0
    if weights[0] > 0 and data.nv2 > 0:
        chi2_v2 = np.sum(((v2_model - data.v2) / data.v2_err) ** 2)
    if weights[1] > 0 and data.nt3amp > 0:
        chi2_t3amp = np.sum(((t3amp_model - data.t3amp) / data.t3amp_err) ** 2)
    if weights[2] > 0 and data.nt3phi > 0:
        chi2_t3phi = np.sum((mod360(t3phi_model - data.t3phi) / data.t3phi_err) ** 2)

    if verbose:
        print(f"V2: {chi2_v2 / max(data.nv2, 1):.4f} ", end="")
        print(f"T3A: {chi2_t3amp / max(data.nt3amp, 1):.4f} ", end="")
        print(f"T3P: {chi2_t3phi / max(data.nt3phi, 1):.4f}")

    return weights[0] * chi2_v2 + weights[1] * chi2_t3amp + weights[2] * chi2_t3phi


__all__ = [
    "chi2_sparco_model",
    "model_complex_visibility",
    "model_observables_from_cvis",
    "normalized_component_image",
    "power_law_flux",
]
