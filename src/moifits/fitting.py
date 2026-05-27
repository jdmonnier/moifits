"""Reusable fitting helpers for parametric OI image models."""

from __future__ import annotations

import numpy as np

from .models import (
    ModelSpec,
    free_parameter_names,
    load_model_spec,
    make_coordinate_grid,
    model_spec_to_dict,
    parameter_bounds,
    parameter_vector,
    render_model,
    render_smooth_disk_ring,
    update_model_parameters,
)


DISK_SPARCO_PARAM_NAMES = [
    "inc",
    "phi",
    "radius",
    "thickness",
    "contrast",
    "sigma",
    "f_star_0",
]


def render_disk(theta, x, y, apply_cutoff=True, normalize_flux=True):
    """
    Render the smooth disk-ring model used by the MCMC bootstrap example.

    theta keys:
        inc, phi, Rin, Rout, contrast, sigma_in, sigma_out
    """
    image = render_smooth_disk_ring(
        x,
        y,
        rin=theta["Rin"],
        rout=theta["Rout"],
        contrast=theta["contrast"],
        sigma_in=theta["sigma_in"],
        sigma_out=theta["sigma_out"],
        inclination_rad=theta["inc"],
        pa_rad=theta["phi"],
        apply_cutoff=apply_cutoff,
    )
    if normalize_flux:
        total = np.sum(image)
        image = np.where(total > 0, image / total, image)
    return image


def unpack_params(z, fixed_params):
    """
    Convert free disk+SPARCO parameters into image and SPARCO parameters.

    z: [inc, phi, radius, thickness, contrast, sigma, f_star_0]
    """
    z = np.asarray(z)
    inc, phi, radius, thickness, contrast, sigma, f_star_0 = z
    rin = radius - thickness / 2.0
    rout = radius + thickness / 2.0

    theta = {
        "inc": float(inc),
        "phi": float(phi),
        "Rin": float(rin),
        "Rout": float(rout),
        "contrast": float(contrast),
        "sigma_in": float(sigma),
        "sigma_out": float(sigma),
    }

    sparco_params = np.array(
        [
            float(f_star_0),
            float(fixed_params["f_bg_0"]),
            float(fixed_params["diameter"]),
            float(fixed_params["d_ind"]),
            float(fixed_params["lambda_0"]),
        ],
        dtype=np.float64,
    )

    return theta, sparco_params


def log_prior_z(z, data, fixed_params):
    """Flat prior for the current disk+SPARCO bootstrap parameterization."""
    del data
    z = np.asarray(z, dtype=float)
    inc, phi, radius, thickness, contrast, sigma, f_star_0 = z

    if not (0.0 <= inc <= 0.9 * np.pi / 2):
        return -np.inf
    if not (0.0 <= phi <= np.pi):
        return -np.inf
    if not (0.01 <= radius <= 10.0):
        return -np.inf
    if not (0.001 <= thickness <= 5.0):
        return -np.inf
    if radius - thickness / 2.0 <= 0:
        return -np.inf
    if not (0.0 <= contrast <= 1.0):
        return -np.inf
    if not (0.001 <= sigma <= 0.5):
        return -np.inf

    f_bg_0 = fixed_params["f_bg_0"]
    if not (0.0 <= f_star_0 <= 0.99):
        return -np.inf
    if f_star_0 + f_bg_0 >= 0.99:
        return -np.inf

    return 0.0


def log_posterior_z(
    z,
    x_grid,
    y_grid,
    ftplan,
    data,
    fixed_params,
    weights=(1.0, 1.0, 1.0),
    apply_cutoff=True,
    normalize_flux=True,
):
    """Log posterior for the current disk+SPARCO bootstrap parameterization."""
    lp = log_prior_z(z, data, fixed_params)
    if not np.isfinite(lp):
        return -np.inf

    theta, sparco_params = unpack_params(z, fixed_params)
    image = render_disk(
        theta,
        x_grid,
        y_grid,
        apply_cutoff=apply_cutoff,
        normalize_flux=normalize_flux,
    )

    try:
        from .oioptimize import chi2_sparco_f

        chi2 = chi2_sparco_f(image, sparco_params, ftplan, data, verbose=False, weights=weights)
    except Exception:
        return -np.inf
    return lp - 0.5 * chi2


def log_prior_model_vector(values: np.ndarray, model: ModelSpec) -> float:
    """Flat prior from free-parameter bounds in a ModelSpec."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        return -np.inf
    for value, bounds in zip(values, parameter_bounds(model)):
        if bounds is None:
            continue
        lower, upper = bounds
        if value < lower or value > upper:
            return -np.inf
    return 0.0


def log_posterior_model_vector(
    values: np.ndarray,
    model: ModelSpec,
    x_grid,
    y_grid,
    ftplan,
    data,
    weights=(1.0, 1.0, 1.0),
) -> float:
    """Log posterior for the generalized multi-component SPARCO model spec."""
    lp = log_prior_model_vector(values, model)
    if not np.isfinite(lp):
        return -np.inf
    trial_model = update_model_parameters(model, values, copy_model=True)
    try:
        from .sparco import chi2_sparco_model

        chi2 = chi2_sparco_model(trial_model, ftplan, data, x_grid, y_grid, weights=weights)
    except Exception:
        return -np.inf
    return lp - 0.5 * chi2


def initial_walkers(z0, bounds, n_walkers, scatter=0.01, seed=42):
    """Initialize emcee walkers around a parameter vector while respecting bounds."""
    rng = np.random.default_rng(seed)
    p0 = np.zeros((n_walkers, z0.size), dtype=float)
    for iparam, value in enumerate(z0):
        bound = bounds[iparam]
        if bound is None:
            sigma = scatter * max(abs(value), 1.0)
            p0[:, iparam] = value + sigma * rng.standard_normal(n_walkers)
            continue

        lower, upper = bound
        if lower is None or upper is None:
            sigma = scatter * max(abs(value), 1.0)
            p0[:, iparam] = value + sigma * rng.standard_normal(n_walkers)
            if lower is not None:
                p0[:, iparam] = np.maximum(p0[:, iparam], lower)
            if upper is not None:
                p0[:, iparam] = np.minimum(p0[:, iparam], upper)
            continue

        width = upper - lower
        if width <= 0:
            raise ValueError(f"invalid bounds for parameter {iparam}: {bound}")
        sigma = scatter * width
        eps = 1e-10 * width
        p0[:, iparam] = value + sigma * rng.standard_normal(n_walkers)
        p0[:, iparam] = np.clip(p0[:, iparam], lower + eps, upper - eps)
    return p0


def posterior_samples(sampler, burn_in, thin):
    """Return chain/log-prob arrays after burn-in and thinning plus flattened views."""
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    chain = chain[burn_in::thin, :, :]
    log_prob = log_prob[burn_in::thin, :]
    flat_chain = chain.reshape(-1, chain.shape[-1])
    flat_log_prob = log_prob.reshape(-1)
    return chain, log_prob, flat_chain, flat_log_prob


def run_emcee_model_fit(
    oifits_file,
    spec_file,
    output_dir=None,
    nx=128,
    pixsize_mas=0.125,
    backend="finufft",
    n_walkers=None,
    n_steps=500,
    burn_in=100,
    thin=1,
    scatter=0.01,
    seed=42,
    weights=(1.0, 0.0, 1.0),
    progress=True,
    **backend_kwargs,
):
    """Run an emcee fit for a JSON ModelSpec and optionally write standard outputs."""
    try:
        import emcee

        from .oichi2 import setup_nfft
        from .readoifits import readoifits
        from .sparco import chi2_sparco_model
    except ModuleNotFoundError as exc:
        if exc.name == "finufft":
            raise ModuleNotFoundError("Install FINUFFT with `python -m pip install -e .`.") from exc
        if exc.name == "cufinufft":
            raise ModuleNotFoundError("Install cuFINUFFT for GPU backend support.") from exc
        if exc.name == "cupy":
            raise ModuleNotFoundError("Install a CUDA-compatible CuPy package for cuFINUFFT backend support.") from exc
        if exc.name == "emcee":
            raise ModuleNotFoundError("Install example dependencies with `python -m pip install -e .[examples]`.") from exc
        raise

    model = load_model_spec(spec_file)
    names = free_parameter_names(model)
    z0 = parameter_vector(model)
    bounds = parameter_bounds(model)

    if z0.size == 0:
        raise ValueError("model spec has no free parameters")
    if n_walkers is None:
        n_walkers = max(32, 2 * z0.size + 2)
    if n_walkers < 2 * z0.size:
        raise ValueError(f"emcee needs at least {2 * z0.size} walkers for {z0.size} parameters")
    if burn_in >= n_steps:
        raise ValueError("burn_in must be smaller than n_steps")
    if thin < 1:
        raise ValueError("thin must be >= 1")

    data = readoifits(str(oifits_file), filter_bad_data=True, redundance_remove=True)
    ftplan = setup_nfft(data, nx, pixsize_mas, backend=backend, **backend_kwargs)
    if backend == "cufinufft":
        ftplan[0]._load_gpu_modules()
    x_grid, y_grid = make_coordinate_grid(nx, pixel_scale=pixsize_mas)

    def log_prob(z):
        return log_posterior_model_vector(
            z,
            model,
            x_grid,
            y_grid,
            ftplan,
            data,
            weights=weights,
        )

    p0 = initial_walkers(z0, bounds, n_walkers=n_walkers, scatter=scatter, seed=seed)
    sampler = emcee.EnsembleSampler(n_walkers, z0.size, log_prob)
    sampler.run_mcmc(p0, n_steps, progress=progress)

    chain, log_prob_chain, flat_chain, flat_log_prob = posterior_samples(sampler, burn_in, thin)
    finite = np.isfinite(flat_log_prob)
    if not np.any(finite):
        raise RuntimeError("sampler produced no finite posterior samples")

    best_idx = np.argmax(np.where(finite, flat_log_prob, -np.inf))
    best_parameters = flat_chain[best_idx]
    median_parameters = np.median(flat_chain[finite], axis=0)
    std_parameters = np.std(flat_chain[finite], axis=0)

    best_model = update_model_parameters(model, best_parameters, copy_model=True)
    chi2 = chi2_sparco_model(best_model, ftplan, data, x_grid, y_grid, weights=weights)
    n_data = 0
    if weights[0] > 0:
        n_data += data.nv2
    if weights[1] > 0:
        n_data += data.nt3amp
    if weights[2] > 0:
        n_data += data.nt3phi
    dof = max(n_data - len(best_parameters), 1)

    image = render_model(best_model, x_grid, y_grid)
    result = {
        "sampler": sampler,
        "model": model,
        "best_model": best_model,
        "parameter_names": names,
        "initial_parameters": z0,
        "best_parameters": best_parameters,
        "median_parameters": median_parameters,
        "std_parameters": std_parameters,
        "bounds": bounds,
        "chain": chain,
        "log_prob": log_prob_chain,
        "acceptance_fraction": sampler.acceptance_fraction,
        "chi2": chi2,
        "reduced_chi2": chi2 / dof,
        "image": image,
        "data": data,
        "ftplan": ftplan,
    }

    if output_dir is not None:
        write_emcee_fit_outputs(result, output_dir, nx=nx, pixsize_mas=pixsize_mas)

    return result


def write_emcee_fit_outputs(result, output_dir, nx, pixsize_mas):
    """Write standard fit products from run_emcee_model_fit."""
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_spec_path = output_dir / "best_model_spec.json"
    with best_spec_path.open("w", encoding="utf-8") as stream:
        json.dump(model_spec_to_dict(result["best_model"]), stream, indent=2)
        stream.write("\n")

    image_path = output_dir / "best_model_image.npy"
    np.save(image_path, result["image"])

    np.savez(
        output_dir / "fit_result.npz",
        parameter_names=np.array(result["parameter_names"]),
        initial_parameters=result["initial_parameters"],
        best_parameters=result["best_parameters"],
        median_parameters=result["median_parameters"],
        std_parameters=result["std_parameters"],
        bounds=np.array(result["bounds"], dtype=object),
        chain=result["chain"],
        log_prob=result["log_prob"],
        acceptance_fraction=result["acceptance_fraction"],
        chi2=result["chi2"],
        reduced_chi2=result["reduced_chi2"],
    )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        extent = 0.5 * nx * pixsize_mas
        im = ax.imshow(
            result["image"],
            origin="lower",
            extent=[-extent, extent, -extent, extent],
            cmap="viridis",
        )
        ax.set_xlabel("RA offset (mas)")
        ax.set_ylabel("Dec offset (mas)")
        ax.set_title("Best-fit model image")
        fig.colorbar(im, ax=ax, label="Normalized intensity")
        fig.tight_layout()
        fig.savefig(output_dir / "best_model_image.png", dpi=150)
        plt.close(fig)

        full_chain = result["sampler"].get_chain()
        names = result["parameter_names"]
        fig, axes = plt.subplots(len(names), 1, figsize=(9, max(2, 1.5 * len(names))), sharex=True)
        if len(names) == 1:
            axes = [axes]
        for iparam, ax in enumerate(axes):
            ax.plot(full_chain[:, :, iparam], color="black", alpha=0.25, linewidth=0.6)
            ax.set_ylabel(names[iparam], fontsize=8)
        axes[-1].set_xlabel("step")
        fig.tight_layout()
        fig.savefig(output_dir / "chains.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Skipping plots: {exc}")

    return {
        "best_spec": best_spec_path,
        "best_image": image_path,
        "fit_result": output_dir / "fit_result.npz",
    }


__all__ = [
    "DISK_SPARCO_PARAM_NAMES",
    "log_posterior_z",
    "log_posterior_model_vector",
    "log_prior_model_vector",
    "log_prior_z",
    "initial_walkers",
    "posterior_samples",
    "render_disk",
    "run_emcee_model_fit",
    "unpack_params",
    "write_emcee_fit_outputs",
]
