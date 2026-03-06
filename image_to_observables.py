"""
Image-based observable generation utilities.

This module bridges a pixel image model to complex visibilities and interferometric
observables on arbitrary UV sampling.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import numpy as np

try:
    from .writeoifits import NoiseConfig, create_oifits_from_model, sample_model_observables
    from .oichi2 import NFFTPlan
except ImportError:  # Allow direct module import from scripts in moifits/testing
    from writeoifits import NoiseConfig, create_oifits_from_model, sample_model_observables
    from oichi2 import NFFTPlan


def image_to_cvis_grid(
    image: np.ndarray,
    pixsize_mas: float,
    ucoord_m: np.ndarray,
    vcoord_m: np.ndarray,
    wavelengths_m: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Evaluate complex visibilities from an image on UV points and wavelengths.

    Args:
        image: 2D image model.
        pixsize_mas: Pixel size in mas.
        ucoord_m: Baseline u coordinates in meters, shape (nrows,).
        vcoord_m: Baseline v coordinates in meters, shape (nrows,).
        wavelengths_m: Wavelength grid in meters, shape (nwave,).
        normalize: Normalize image flux before FT.

    Returns:
        cvis array with shape (nrows, nwave).
    """
    ucoord_m = np.asarray(ucoord_m, dtype=float)
    vcoord_m = np.asarray(vcoord_m, dtype=float)
    wavelengths_m = np.asarray(wavelengths_m, dtype=float)
    if ucoord_m.shape != vcoord_m.shape:
        raise ValueError("ucoord_m and vcoord_m must have the same shape")

    image_cvis_fn = make_image_cvis_model(
        image=image,
        pixsize_mas=pixsize_mas,
        normalize=normalize,
    )
    cvis = np.zeros((ucoord_m.size, wavelengths_m.size), dtype=np.complex128)
    for iw, lam in enumerate(wavelengths_m):
        cvis[:, iw] = image_cvis_fn(ucoord_m / lam, vcoord_m / lam, float(lam))
    return cvis


def make_image_cvis_model(
    image: np.ndarray,
    pixsize_mas: float,
    normalize: bool = True,
) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    Build a model callable compatible with create_oifits.sample_model_observables.

    Args:
        image: 2D image array (or flattened square image).
        pixsize_mas: Pixel size in milliarcseconds.
        normalize: Normalize image flux to 1 before FT.

    Returns:
        image_cvis_fn(u_lam, v_lam, wavelength_m) -> complex visibility array
    """
    img = np.asarray(image, dtype=float)
    if img.ndim == 1:
        nx = int(np.sqrt(img.size))
        if nx * nx != img.size:
            raise ValueError("Flattened image length must be a perfect square")
        img = img.reshape(nx, nx)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError("image must be a square 2D array")

    model_image = img / np.sum(img) if normalize else img.copy()
    nx = model_image.shape[0]

    def image_cvis_fn(u_lam: np.ndarray, v_lam: np.ndarray, wavelength_m: float) -> np.ndarray:
        del wavelength_m  # Included for signature compatibility.
        uv = np.vstack([np.asarray(u_lam, dtype=float), np.asarray(v_lam, dtype=float)])
        plan = NFFTPlan(uv, nx, pixsize_mas)
        return plan.forward(model_image)

    return image_cvis_fn


def sample_image_observables(
    image: np.ndarray,
    pixsize_mas: float,
    sampling: Dict[str, np.ndarray],
    wavelengths_m: np.ndarray,
    noise: NoiseConfig = NoiseConfig(),
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute VIS/VIS2/T3 arrays from an image on provided UV sampling.

    Args:
        image: 2D image model.
        pixsize_mas: Pixel size in mas.
        sampling: Output of create_oifits.generate_uv_sampling().
        wavelengths_m: Effective wavelengths in meters.
        noise: Error/noise settings.
        normalize: Normalize image flux before FT.

    Returns:
        Observable dictionary compatible with create_oifits writers.
    """
    image_cvis_fn = make_image_cvis_model(
        image=image,
        pixsize_mas=pixsize_mas,
        normalize=normalize,
    )
    return sample_model_observables(
        model_cvis=image_cvis_fn,
        sampling=sampling,
        wavelengths_m=wavelengths_m,
        noise=noise,
    )


def create_oifits_from_image(
    output_path: str,
    image: np.ndarray,
    pixsize_mas: float,
    station_enu_m: np.ndarray,
    hour_angles_rad: Iterable[float],
    dec_rad: float,
    wavelengths_m: Iterable[float],
    eff_band_m: Iterable[float] | None = None,
    target_name: str = "SYNTH_TARGET",
    target_id: int = 1,
    insname: str = "SYNTH_INS",
    mjd_start: float = 60000.0,
    cadence_sec: float = 600.0,
    noise: NoiseConfig = NoiseConfig(),
    normalize: bool = True,
    overwrite: bool = True,
) -> str:
    """
    Convenience wrapper: create synthetic OIFITS directly from an image.
    """
    image_cvis_fn = make_image_cvis_model(
        image=image,
        pixsize_mas=pixsize_mas,
        normalize=normalize,
    )
    return create_oifits_from_model(
        output_path=output_path,
        model_cvis=image_cvis_fn,
        station_enu_m=station_enu_m,
        hour_angles_rad=hour_angles_rad,
        dec_rad=dec_rad,
        wavelengths_m=wavelengths_m,
        eff_band_m=eff_band_m,
        target_name=target_name,
        target_id=target_id,
        insname=insname,
        mjd_start=mjd_start,
        cadence_sec=cadence_sec,
        noise=noise,
        overwrite=overwrite,
    )


__all__ = [
    "image_to_cvis_grid",
    "make_image_cvis_model",
    "sample_image_observables",
    "create_oifits_from_image",
]
