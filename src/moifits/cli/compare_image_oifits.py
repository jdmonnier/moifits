#!/usr/bin/env python3
"""
Compare an image-based model against OIFITS observables.

Model flux split:
    star: point source (V=1) with power-law spectral index
    halo: resolved/incoherent by default (V=0) with power-law spectral index
    image: resolved component from input image with power-law spectral index

At the reference wavelength lambda0, the image fraction is:
    f_img_0 = 1 - f_star_0 - f_halo_0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from moifits.oichi2 import NFFTPlan, mod360, vis_to_t3, vis_to_v2
from moifits.plot_oifits import plot_observables_model_comparison
from moifits.readoifits import readoifits


def _power_law_flux(frac0: float, spectral_index: float, lam: np.ndarray, lambda0: float) -> np.ndarray:
    return float(frac0) * (np.asarray(lam, dtype=float) / float(lambda0)) ** float(spectral_index)


def _load_image(path: Path) -> np.ndarray:
    image = np.load(path).astype(float)
    if image.ndim == 1:
        nx = int(np.sqrt(image.size))
        if nx * nx != image.size:
            raise ValueError("Flattened image length must be a perfect square.")
        image = image.reshape(nx, nx)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Input image must be a square 2D array.")
    total = np.sum(image)
    if total <= 0:
        raise ValueError("Input image total flux must be > 0.")
    return image / total


def _compute_model_cvis(
    image: np.ndarray,
    data,
    pixsize_mas: float,
    backend: str,
    eps: float,
    f_star_0: float,
    f_halo_0: float,
    star_index: float,
    halo_index: float,
    halo_visibility: float,
    image_index: float,
    lambda0_m: float,
    gpu_device_id: int | None,
) -> np.ndarray:
    backend_kwargs = {}
    if gpu_device_id is not None:
        backend_kwargs["gpu_device_id"] = gpu_device_id
    plan = NFFTPlan(data.uv, image.shape[0], pixsize_mas, backend=backend, eps=eps, **backend_kwargs)
    cvis_img = plan.forward(image)

    lam = np.asarray(data.uv_lam, dtype=float)
    f_img_0 = 1.0 - f_star_0 - f_halo_0
    f_star = _power_law_flux(f_star_0, star_index, lam, lambda0_m)
    f_halo = _power_law_flux(f_halo_0, halo_index, lam, lambda0_m)
    f_img = _power_law_flux(f_img_0, image_index, lam, lambda0_m)
    denom = f_star + f_halo + f_img
    if np.any(denom == 0):
        raise ValueError("Model denominator has zero values. Check fractions and spectral indices.")

    # Star is point-like by default; halo defaults to resolved/incoherent via V=0.
    return (f_star + halo_visibility * f_halo + f_img * cvis_img) / denom


def _observable_chi2(data_val: np.ndarray, model_val: np.ndarray, err: np.ndarray, angle: bool = False) -> tuple[float, np.ndarray]:
    if angle:
        resid = mod360(model_val - data_val)
    else:
        resid = model_val - data_val
    chi2 = float(np.sum((resid / err) ** 2))
    return chi2, resid


def _pick_plot_wavelengths(lam_values: np.ndarray, requested_um: Optional[list[float]], n_default: int) -> np.ndarray:
    uniq = np.unique(np.asarray(lam_values, dtype=float))
    if uniq.size == 0:
        return uniq
    if requested_um:
        req_m = np.asarray(requested_um, dtype=float) * 1e-6
        chosen = []
        for w in req_m:
            idx = int(np.argmin(np.abs(uniq - w)))
            chosen.append(uniq[idx])
        return np.unique(np.asarray(chosen))
    n = max(1, min(int(n_default), uniq.size))
    idx = np.rint(np.linspace(0, uniq.size - 1, n)).astype(int)
    return np.unique(uniq[idx])


def _compute_dense_v2_curves(
    image: np.ndarray,
    data,
    pixsize_mas: float,
    backend: str,
    eps: float,
    f_star_0: float,
    f_halo_0: float,
    star_index: float,
    halo_index: float,
    halo_visibility: float,
    image_index: float,
    lambda0_m: float,
    gpu_device_id: int | None,
    selected_lam_m: np.ndarray,
    n_baseline_dense: int = 512,
    n_angle: int = 180,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for lam in selected_lam_m:
        m = np.asarray(data.v2_lam) == lam
        if not np.any(m):
            continue
        b = np.asarray(data.v2_baseline)[m]
        b = b[np.isfinite(b)]
        if b.size < 2:
            continue
        bmin = float(np.min(b))
        bmax = float(np.max(b))
        if bmax <= bmin:
            continue
        b_dense = np.linspace(bmin, bmax, n_baseline_dense)
        theta = np.linspace(0.0, 2.0 * np.pi, n_angle, endpoint=False)
        uu = (b_dense[:, None] * np.cos(theta)[None, :]).reshape(-1)
        vv = (b_dense[:, None] * np.sin(theta)[None, :]).reshape(-1)
        uv = np.vstack([uu, vv])
        lam_arr = np.full(uv.shape[1], float(lam), dtype=float)

        class _Tmp:
            pass

        tmp = _Tmp()
        tmp.uv = uv
        tmp.uv_lam = lam_arr
        cvis = _compute_model_cvis(
            image=image,
            data=tmp,
            pixsize_mas=pixsize_mas,
            backend=backend,
            eps=eps,
            f_star_0=f_star_0,
            f_halo_0=f_halo_0,
            star_index=star_index,
            halo_index=halo_index,
            halo_visibility=halo_visibility,
            image_index=image_index,
            lambda0_m=lambda0_m,
            gpu_device_id=gpu_device_id,
        )
        v2 = np.abs(cvis) ** 2
        v2_grid = v2.reshape(n_baseline_dense, n_angle)
        v2_radial = np.nanmean(v2_grid, axis=1)
        curves[float(lam)] = (b_dense, v2_radial)
    return curves


def compare_image_to_oifits(
    oifits_file: Path,
    image_file: Path,
    pixsize_mas: float,
    save: Path | None,
    backend: str,
    eps: float,
    f_star_0: float,
    f_halo_0: float,
    star_index: float,
    halo_index: float,
    halo_visibility: float,
    image_index: float,
    lambda0_m: float,
    gpu_device_id: int | None,
    no_filter: bool,
    no_redundance_remove: bool,
    plot_wavelengths_um: Optional[list[float]] = None,
    n_plot_wavelengths: int = 5,
):
    image = _load_image(image_file)
    data = readoifits(
        str(oifits_file),
        filter_bad_data=not no_filter,
        redundance_remove=not no_redundance_remove,
    )

    cvis_model = _compute_model_cvis(
        image=image,
        data=data,
        pixsize_mas=pixsize_mas,
        backend=backend,
        eps=eps,
        f_star_0=f_star_0,
        f_halo_0=f_halo_0,
        star_index=star_index,
        halo_index=halo_index,
        halo_visibility=halo_visibility,
        image_index=image_index,
        lambda0_m=lambda0_m,
        gpu_device_id=gpu_device_id,
    )

    v2_model = vis_to_v2(cvis_model, data.indx_v2)
    _, _, t3phi_model = vis_to_t3(cvis_model, data.indx_t3_1, data.indx_t3_2, data.indx_t3_3)
    selected_lam_m = _pick_plot_wavelengths(np.asarray(data.v2_lam), plot_wavelengths_um, n_plot_wavelengths)
    dense_v2_curves = _compute_dense_v2_curves(
        image=image,
        data=data,
        pixsize_mas=pixsize_mas,
        backend=backend,
        eps=eps,
        f_star_0=f_star_0,
        f_halo_0=f_halo_0,
        star_index=star_index,
        halo_index=halo_index,
        halo_visibility=halo_visibility,
        image_index=image_index,
        lambda0_m=lambda0_m,
        gpu_device_id=gpu_device_id,
        selected_lam_m=selected_lam_m,
    )

    chi2_v2, _ = _observable_chi2(data.v2, v2_model, data.v2_err, angle=False)
    chi2_t3phi, _ = _observable_chi2(data.t3phi, t3phi_model, data.t3phi_err, angle=True)

    ndof = max(data.nv2 + data.nt3phi - 0, 1)
    chi2_total = chi2_v2 + chi2_t3phi
    red_chi2 = chi2_total / ndof

    fig, _ = plot_observables_model_comparison(
        data,
        v2_model=v2_model,
        t3phi_model=t3phi_model,
        figsize=(13, 8),
        selected_wavelengths_m=None if not plot_wavelengths_um else [w * 1e-6 for w in plot_wavelengths_um],
        n_wavelength_lines=n_plot_wavelengths,
        dense_v2_curves=dense_v2_curves,
    )

    fig.suptitle(
        f"{oifits_file.name}\n"
        f"chi2_tot={chi2_total:.3g} red_chi2={red_chi2:.3g} "
        f"(v2={chi2_v2:.3g}, t3p={chi2_t3phi:.3g})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    print(f"chi2_v2    = {chi2_v2:.6g}")
    print(f"chi2_t3phi = {chi2_t3phi:.6g}")
    print(f"chi2_total = {chi2_total:.6g}")
    print(f"red_chi2   = {red_chi2:.6g}")

    if save is not None:
        fig.savefig(save, dpi=170, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare an image model against OIFITS observables.")
    parser.add_argument("oifits_file", type=Path, help="Path to OIFITS file.")
    parser.add_argument("image_file", type=Path, help="Path to .npy image file (square image).")
    parser.add_argument("--pixsize", type=float, required=True, help="Image pixel scale in mas.")
    parser.add_argument(
        "--backend",
        choices=["finufft", "cufinufft", "direct"],
        default="finufft",
        help="Fourier backend.",
    )
    parser.add_argument("--gpu-device-id", type=int, default=None, help="CUDA GPU ID for cufinufft.")
    parser.add_argument("--eps", type=float, default=1e-12, help="NFFT tolerance.")
    parser.add_argument("--lambda0", type=float, default=1.65e-6, help="Reference wavelength in meters.")
    parser.add_argument("--f-star", type=float, default=0.0, help="Star flux fraction at lambda0.")
    parser.add_argument("--f-halo", type=float, default=0.0, help="Halo flux fraction at lambda0.")
    parser.add_argument("--star-index", type=float, default=-4.0, help="Star spectral power-law index.")
    parser.add_argument("--halo-index", type=float, default=0.0, help="Halo spectral power-law index.")
    parser.add_argument(
        "--halo-visibility",
        type=float,
        default=0.0,
        help="Halo visibility factor: 0 for resolved/incoherent, 1 for unresolved point source.",
    )
    parser.add_argument("--image-index", type=float, default=0.0, help="Image spectral power-law index.")
    parser.add_argument(
        "--plot-wavelengths",
        type=float,
        nargs="+",
        default=None,
        help="Explicit wavelengths (um) for model lines. If omitted, uses --n-plot-wavelengths equally spaced channels.",
    )
    parser.add_argument(
        "--n-plot-wavelengths",
        type=int,
        default=5,
        help="Number of equally spaced wavelengths for model lines when --plot-wavelengths is not provided.",
    )
    parser.add_argument("--save", type=Path, default=None, help="Output plot path. If omitted, opens interactive window.")
    parser.add_argument("--no-filter", action="store_true", help="Disable bad-data filtering while reading.")
    parser.add_argument("--no-redundance-remove", action="store_true", help="Disable redundant UV point removal while reading.")
    args = parser.parse_args()

    if args.f_star < 0 or args.f_halo < 0:
        raise SystemExit("f-star and f-halo must be >= 0.")
    if args.f_star + args.f_halo >= 1.0:
        raise SystemExit("f-star + f-halo must be < 1.0 so the image keeps positive flux fraction.")
    if args.n_plot_wavelengths < 1:
        raise SystemExit("--n-plot-wavelengths must be >= 1.")

    compare_image_to_oifits(
        oifits_file=args.oifits_file,
        image_file=args.image_file,
        pixsize_mas=args.pixsize,
        save=args.save,
        backend=args.backend,
        eps=args.eps,
        f_star_0=args.f_star,
        f_halo_0=args.f_halo,
        star_index=args.star_index,
        halo_index=args.halo_index,
        halo_visibility=args.halo_visibility,
        image_index=args.image_index,
        lambda0_m=args.lambda0,
        gpu_device_id=args.gpu_device_id,
        no_filter=args.no_filter,
        no_redundance_remove=args.no_redundance_remove,
        plot_wavelengths_um=args.plot_wavelengths,
        n_plot_wavelengths=args.n_plot_wavelengths,
    )


if __name__ == "__main__":
    main()
