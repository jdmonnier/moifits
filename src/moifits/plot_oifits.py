"""
Plotting helpers for OIData products.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

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


def _plot_model_data_pair(
    axes_pair,
    x: np.ndarray,
    lam: np.ndarray,
    line_lam: np.ndarray,
    y_data: np.ndarray,
    y_model: np.ndarray,
    yerr: np.ndarray,
    lam_to_color: dict[float, tuple[float, float, float, float]],
    title: str,
    ylabel: str,
    resid_label: str,
    dense_model_curves: Optional[dict[float, tuple[np.ndarray, np.ndarray]]] = None,
    marker_size: float = 24,
    alpha: float = 0.7,
):
    ax_main, ax_resid = axes_pair
    x = np.asarray(x)
    lam = np.asarray(lam)
    y_data = np.asarray(y_data)
    y_model = np.asarray(y_model)
    yerr = np.asarray(yerr)

    ax_main.errorbar(
        x,
        y_data,
        yerr=yerr,
        fmt="o",
        markersize=np.sqrt(marker_size),
        alpha=alpha,
        linewidth=0.8,
        capsize=0,
        label="OIFITS",
        zorder=2,
    )
    uniq_lam = np.unique(line_lam)
    for i, lval in enumerate(uniq_lam):
        color = lam_to_color[float(lval)]
        label = f"Model {lval*1e6:.2f} um"
        if dense_model_curves is not None and float(lval) in dense_model_curves:
            x_dense, y_dense = dense_model_curves[float(lval)]
            ax_main.plot(x_dense, y_dense, color=color, linewidth=1.8, alpha=0.95, label=label, zorder=3)
        else:
            m = lam == lval
            if not np.any(m):
                continue
            order = np.argsort(x[m])
            ax_main.plot(x[m][order], y_model[m][order], color=color, linewidth=1.2, alpha=0.9, label=label, zorder=3)

    ax_main.set_title(title)
    ax_main.set_ylabel(ylabel)
    ax_main.legend(loc="best")

    resid = y_model - y_data
    z = resid / yerr
    ax_resid.scatter(x, z, s=10, alpha=0.8, color="0.2", zorder=2)
    ax_resid.axhspan(-1.0, 1.0, color="0.92", zorder=0)
    ax_resid.axhline(0.0, color="0.35", linewidth=1.0)
    ax_resid.axhline(3.0, color="0.75", linewidth=0.8, linestyle="--")
    ax_resid.axhline(-3.0, color="0.75", linewidth=0.8, linestyle="--")
    finite = np.isfinite(z)
    if np.any(finite):
        zmax = np.nanpercentile(np.abs(z[finite]), 95)
        ax_resid.set_ylim(-max(3.5, 1.1 * zmax), max(3.5, 1.1 * zmax))
    ax_resid.set_ylabel(resid_label)


def _pick_plot_wavelengths(
    lam_values: np.ndarray,
    selected_wavelengths_m: Optional[Sequence[float]],
    n_wavelength_lines: int,
) -> np.ndarray:
    uniq = np.unique(np.asarray(lam_values))
    if uniq.size == 0:
        return uniq
    if selected_wavelengths_m is not None and len(selected_wavelengths_m) > 0:
        requested = np.asarray(selected_wavelengths_m, dtype=float)
        chosen = []
        for w in requested:
            idx = int(np.argmin(np.abs(uniq - w)))
            chosen.append(uniq[idx])
        return np.unique(np.asarray(chosen))

    n = max(1, min(int(n_wavelength_lines), uniq.size))
    idx = np.rint(np.linspace(0, uniq.size - 1, n)).astype(int)
    return np.unique(uniq[idx])


def _make_lambda_color_map(line_lam: np.ndarray) -> dict[float, tuple[float, float, float, float]]:
    uniq_lam = np.unique(np.asarray(line_lam))
    cmap = plt.get_cmap("viridis")
    out: dict[float, tuple[float, float, float, float]] = {}
    for i, lval in enumerate(uniq_lam):
        out[float(lval)] = cmap(i / max(len(uniq_lam) - 1, 1))
    return out


def _wrapped_deg(phi_deg: np.ndarray) -> np.ndarray:
    return ((np.asarray(phi_deg) + 180.0) % 360.0) - 180.0


def plot_observables_model_comparison(
    data,
    v2_model: np.ndarray,
    t3phi_model: np.ndarray,
    figsize=(12, 7),
    selected_wavelengths_m: Optional[Sequence[float]] = None,
    n_wavelength_lines: int = 5,
    dense_v2_curves: Optional[dict[float, tuple[np.ndarray, np.ndarray]]] = None,
):
    """
    Plot model-vs-data and normalized residuals for V2 and T3PHI.

    Layout:
      row 1: V2 (main, residual)
      row 2: T3PHI (main, residual)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex="col")

    v2_line_lam = _pick_plot_wavelengths(np.asarray(data.v2_lam), selected_wavelengths_m, n_wavelength_lines)
    t3_line_lam = _pick_plot_wavelengths(np.asarray(data.t3_lam), selected_wavelengths_m, n_wavelength_lines)
    combined_line_lam = np.unique(np.concatenate([v2_line_lam, t3_line_lam]))
    lam_to_color = _make_lambda_color_map(combined_line_lam)

    _plot_model_data_pair(
        axes[0],
        np.asarray(data.v2_baseline),
        np.asarray(data.v2_lam),
        v2_line_lam,
        np.asarray(data.v2),
        np.asarray(v2_model),
        np.asarray(data.v2_err),
        lam_to_color=lam_to_color,
        dense_model_curves=dense_v2_curves,
        title="V2: Data vs Model",
        ylabel="V2",
        resid_label="(model-data)/sigma",
    )
    axes[0, 0].set_xlabel("Baseline (lambda^-1)")
    axes[0, 1].set_xlabel("Baseline (lambda^-1)")

    # Closure phase residual uses wrapped angle difference.
    wrapped = _wrapped_deg(np.asarray(t3phi_model) - np.asarray(data.t3phi))
    z_t3phi = wrapped / np.asarray(data.t3phi_err)
    axes[1, 0].errorbar(
        np.asarray(data.t3_maxbaseline),
        _wrapped_deg(np.asarray(data.t3phi)),
        yerr=np.asarray(data.t3phi_err),
        fmt="o",
        markersize=np.sqrt(24),
        alpha=0.7,
        linewidth=0.8,
        capsize=0,
        label="OIFITS",
        zorder=2,
    )
    uniq_lam = t3_line_lam
    for i, lval in enumerate(uniq_lam):
        m = np.asarray(data.t3_lam) == lval
        if not np.any(m):
            continue
        color = lam_to_color[float(lval)]
        label = f"Model {lval*1e6:.2f} um"
        axes[1, 0].scatter(
            np.asarray(data.t3_maxbaseline)[m],
            _wrapped_deg(np.asarray(t3phi_model)[m]),
            s=16,
            alpha=0.9,
            color=color,
            label=label,
            zorder=3,
        )

    axes[1, 0].set_title("T3PHI: Data vs Model")
    axes[1, 0].set_ylabel("T3PHI (deg)")
    axes[1, 0].set_ylim(-190, 190)
    axes[1, 0].set_yticks([-180, -120, -60, 0, 60, 120, 180])
    axes[1, 0].set_xlabel("Max Triangle Baseline (lambda^-1)")
    if axes[1, 0].get_legend() is not None:
        axes[1, 0].get_legend().remove()
    axes[1, 1].scatter(
        np.asarray(data.t3_maxbaseline),
        z_t3phi,
        s=10,
        alpha=0.8,
        color="0.2",
        zorder=2,
    )
    axes[1, 1].axhspan(-1.0, 1.0, color="0.92", zorder=0)
    axes[1, 1].axhline(0.0, color="0.35", linewidth=1.0)
    axes[1, 1].axhline(3.0, color="0.75", linewidth=0.8, linestyle="--")
    axes[1, 1].axhline(-3.0, color="0.75", linewidth=0.8, linestyle="--")
    finite = np.isfinite(z_t3phi)
    if np.any(finite):
        zmax = np.nanpercentile(np.abs(z_t3phi[finite]), 95)
        axes[1, 1].set_ylim(-max(3.5, 1.1 * zmax), max(3.5, 1.1 * zmax))
    axes[1, 1].set_ylabel("(model-data)/sigma")
    axes[1, 1].set_xlabel("Max Triangle Baseline (lambda^-1)")

    # Shared compact legend: OIFITS markers + wavelength-specific model lines.
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="0.25", markersize=5, label="OIFITS"),
    ]
    for lval in combined_line_lam:
        handles.append(
            Line2D(
                [0],
                [0],
                color=lam_to_color[float(lval)],
                linewidth=2.0,
                label=f"Model {lval*1e6:.2f} um",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(handles), 6),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=1.8,
        columnspacing=0.9,
    )

    fig.tight_layout()
    return fig, axes


__all__ = [
    "plot_vis_vs_baseline",
    "plot_vis2_vs_baseline",
    "plot_t3_vs_baseline",
    "plot_uv_coverage",
    "plot_observables_overview",
    "plot_observables_model_comparison",
]
