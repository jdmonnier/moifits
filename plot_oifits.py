"""
Plotting helpers for OIData products.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _get_color_values(
    color_by: Optional[str],
    wavelengths_m: np.ndarray,
    mjd: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if color_by is None:
        return None, None
    if color_by == "wavelength":
        return wavelengths_m * 1e6, "Wavelength (um)"
    if color_by == "mjd":
        return mjd, "MJD"
    raise ValueError("color_by must be one of: None, 'wavelength', 'mjd'")


def _scatter_observable(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray],
    color_values: Optional[np.ndarray],
    color_label: Optional[str],
    title: str,
    ylabel: str,
    show_errors: bool,
    cmap: str,
    marker_size: float,
    alpha: float,
):
    if x.size == 0:
        ax.set_title(title)
        ax.set_xlabel("Baseline (lambda^-1)")
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    if show_errors and yerr is not None:
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor="0.45",
            elinewidth=0.8,
            capsize=1.8,
            alpha=0.6,
            zorder=1,
        )

    if color_values is None:
        ax.scatter(x, y, s=marker_size, alpha=alpha, zorder=2)
    else:
        sc = ax.scatter(x, y, c=color_values, cmap=cmap, s=marker_size, alpha=alpha, zorder=2)
        cbar = ax.figure.colorbar(sc, ax=ax)
        cbar.set_label(color_label)

    ax.set_title(title)
    ax.set_xlabel("Baseline (lambda^-1)")
    ax.set_ylabel(ylabel)


def plot_vis_vs_baseline(
    data,
    quantity: str = "amp",
    ax=None,
    color_by: Optional[str] = "wavelength",
    show_errors: bool = True,
    cmap: str = "viridis",
    marker_size: float = 12,
    alpha: float = 0.85,
):
    """
    Plot VIS amplitude or phase versus baseline.

    quantity: "amp" or "phi"
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if quantity == "amp":
        y = data.visamp
        yerr = data.visamp_err if show_errors else None
        ylabel = "VISAMP"
        title = "VIS Amplitude vs Baseline"
    elif quantity == "phi":
        y = data.visphi
        yerr = data.visphi_err if show_errors else None
        ylabel = "VISPHI (deg)"
        title = "VIS Phase vs Baseline"
    else:
        raise ValueError("quantity must be 'amp' or 'phi'")

    color_values, color_label = _get_color_values(color_by, data.vis_lam, data.vis_mjd)
    _scatter_observable(
        ax=ax,
        x=np.asarray(data.vis_baseline),
        y=np.asarray(y),
        yerr=np.asarray(yerr) if yerr is not None else None,
        color_values=color_values,
        color_label=color_label,
        title=title,
        ylabel=ylabel,
        show_errors=show_errors,
        cmap=cmap,
        marker_size=marker_size,
        alpha=alpha,
    )
    return ax


def plot_vis2_vs_baseline(
    data,
    ax=None,
    color_by: Optional[str] = "wavelength",
    show_errors: bool = True,
    cmap: str = "viridis",
    marker_size: float = 12,
    alpha: float = 0.85,
):
    """Plot VIS2 versus baseline."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    color_values, color_label = _get_color_values(color_by, data.v2_lam, data.v2_mjd)
    _scatter_observable(
        ax=ax,
        x=np.asarray(data.v2_baseline),
        y=np.asarray(data.v2),
        yerr=np.asarray(data.v2_err) if show_errors else None,
        color_values=color_values,
        color_label=color_label,
        title="VIS2 vs Baseline",
        ylabel="VIS2",
        show_errors=show_errors,
        cmap=cmap,
        marker_size=marker_size,
        alpha=alpha,
    )
    return ax


def plot_t3_vs_baseline(
    data,
    quantity: str = "phi",
    use_max_baseline: bool = True,
    ax=None,
    color_by: Optional[str] = "wavelength",
    show_errors: bool = True,
    cmap: str = "viridis",
    marker_size: float = 12,
    alpha: float = 0.85,
):
    """
    Plot T3 amplitude or closure phase versus baseline.

    quantity: "amp" or "phi"
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    baseline = np.asarray(data.t3_maxbaseline if use_max_baseline else data.t3_baseline)
    x_label = "Max Triangle Baseline (lambda^-1)" if use_max_baseline else "Triangle Geometric Baseline (lambda^-1)"

    if quantity == "amp":
        y = data.t3amp
        yerr = data.t3amp_err if show_errors else None
        ylabel = "T3AMP"
        title = "T3 Amplitude vs Baseline"
    elif quantity == "phi":
        y = data.t3phi
        yerr = data.t3phi_err if show_errors else None
        ylabel = "T3PHI (deg)"
        title = "Closure Phase (T3PHI) vs Baseline"
    else:
        raise ValueError("quantity must be 'amp' or 'phi'")

    color_values, color_label = _get_color_values(color_by, data.t3_lam, data.t3_mjd)
    _scatter_observable(
        ax=ax,
        x=baseline,
        y=np.asarray(y),
        yerr=np.asarray(yerr) if yerr is not None else None,
        color_values=color_values,
        color_label=color_label,
        title=title,
        ylabel=ylabel,
        show_errors=show_errors,
        cmap=cmap,
        marker_size=marker_size,
        alpha=alpha,
    )
    ax.set_xlabel(x_label)
    return ax


def plot_uv_coverage(
    data,
    ax=None,
    color_by: Optional[str] = "wavelength",
    show_conjugate: bool = True,
    cmap: str = "viridis",
    marker_size: float = 10,
    alpha: float = 0.85,
):
    """Plot UV coverage using data.uv."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    u = np.asarray(data.uv[0]) if data.uv.size > 0 else np.array([])
    v = np.asarray(data.uv[1]) if data.uv.size > 0 else np.array([])

    if u.size == 0:
        ax.set_title("UV Coverage")
        ax.text(0.5, 0.5, "No UV points", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("u (lambda^-1)")
        ax.set_ylabel("v (lambda^-1)")
        ax.set_aspect("equal", adjustable="box")
        return ax

    color_values, color_label = _get_color_values(color_by, data.uv_lam, data.uv_mjd)

    if color_values is None:
        ax.scatter(u, v, s=marker_size, alpha=alpha, label="+(u,v)")
        if show_conjugate:
            ax.scatter(-u, -v, s=marker_size, alpha=alpha * 0.6, label="-(u,v)")
    else:
        sc = ax.scatter(u, v, c=color_values, cmap=cmap, s=marker_size, alpha=alpha, label="+(u,v)")
        if show_conjugate:
            ax.scatter(-u, -v, c=color_values, cmap=cmap, s=marker_size, alpha=alpha * 0.6, label="-(u,v)")
        cbar = ax.figure.colorbar(sc, ax=ax)
        cbar.set_label(color_label)

    ax.set_title("UV Coverage")
    ax.set_xlabel("u (lambda^-1)")
    ax.set_ylabel("v (lambda^-1)")
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, color="0.85", linewidth=1.0)
    ax.axvline(0.0, color="0.85", linewidth=1.0)
    ax.legend(loc="best")
    return ax


def plot_observables_overview(
    data,
    color_by: Optional[str] = "wavelength",
    show_errors: bool = True,
    show_conjugate_uv: bool = True,
    figsize=(12, 10),
):
    """
    Plot a 2x2 overview: VIS2, T3PHI, VISAMP, UV coverage.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plot_vis2_vs_baseline(data, ax=axes[0, 0], color_by=color_by, show_errors=show_errors)
    plot_t3_vs_baseline(data, quantity="phi", ax=axes[0, 1], color_by=color_by, show_errors=show_errors)
    plot_vis_vs_baseline(data, quantity="amp", ax=axes[1, 0], color_by=color_by, show_errors=show_errors)
    plot_uv_coverage(data, ax=axes[1, 1], color_by=color_by, show_conjugate=show_conjugate_uv)
    fig.tight_layout()
    return fig, axes


__all__ = [
    "plot_vis_vs_baseline",
    "plot_vis2_vs_baseline",
    "plot_t3_vs_baseline",
    "plot_uv_coverage",
    "plot_observables_overview",
]
