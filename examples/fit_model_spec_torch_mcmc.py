#!/usr/bin/env python3
"""
Fit a JSON model spec with a PyTorch random-walk MCMC sampler.

This example uses a direct DFT likelihood implemented in torch, so it can run
on CUDA or MPS for small images and moderate OIFITS data sets. It is intended as
a GPU-capable sampler example, not as a replacement for the faster FINUFFT path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Allow running this example directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moifits.models import (
    free_parameter_names,
    load_model_spec,
    make_coordinate_grid,
    model_spec_to_dict,
    parameter_bounds,
    parameter_vector,
    render_model,
    update_model_parameters,
)


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency 'torch'. Install the package with `python -m pip install -e .`.") from exc
    return torch


def _select_device(device_name: str):
    torch = _import_torch()
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def _spec_value(spec, full_name, name_to_index, z, n_chains, device, dtype):
    torch = _import_torch()
    if full_name in name_to_index:
        return z[:, name_to_index[full_name]]
    return torch.full((n_chains,), float(spec.value), device=device, dtype=dtype)


def _component_value(component, name, default, name_to_index, z, n_chains, device, dtype):
    from moifits.models import ParameterSpec

    spec = component.params.get(name, ParameterSpec(default, fixed=True))
    return _spec_value(spec, f"{component.name}.{name}", name_to_index, z, n_chains, device, dtype)


def _rotated_offsets(x_grid, y_grid, x0, y0, pa_deg):
    torch = _import_torch()
    theta = torch.deg2rad(pa_deg)[:, None, None]
    dx = x_grid[None, :, :] - x0[:, None, None]
    dy = y_grid[None, :, :] - y0[:, None, None]
    x_rot = dx * torch.cos(theta) + dy * torch.sin(theta)
    y_rot = -dx * torch.sin(theta) + dy * torch.cos(theta)
    return x_rot, y_rot


def _render_component_torch(component, x_grid, y_grid, name_to_index, z, device, dtype):
    torch = _import_torch()
    n_chains = z.shape[0]
    kind = component.kind.replace("_", "-").lower()
    x0 = _component_value(component, "x0", 0.0, name_to_index, z, n_chains, device, dtype)
    y0 = _component_value(component, "y0", 0.0, name_to_index, z, n_chains, device, dtype)
    pa = _component_value(component, "pa_deg", 0.0, name_to_index, z, n_chains, device, dtype)
    x_rot, y_rot = _rotated_offsets(x_grid, y_grid, x0, y0, pa)

    if kind == "gaussian":
        fwhm_major = _component_value(component, "fwhm_major", 10.0, name_to_index, z, n_chains, device, dtype)
        fwhm_minor = _component_value(component, "fwhm_minor", 10.0, name_to_index, z, n_chains, device, dtype)
        sigma_major = fwhm_major[:, None, None] / 2.3548200450309493
        sigma_minor = fwhm_minor[:, None, None] / 2.3548200450309493
        image = torch.exp(-0.5 * ((x_rot / sigma_major) ** 2 + (y_rot / sigma_minor) ** 2))
    elif kind in {"ring", "elliptical-ring"}:
        radius = _component_value(component, "radius", 1.0, name_to_index, z, n_chains, device, dtype)
        width = _component_value(component, "width", 0.5, name_to_index, z, n_chains, device, dtype)
        axis_ratio = _component_value(component, "axis_ratio", 1.0, name_to_index, z, n_chains, device, dtype)
        elliptical_radius = torch.sqrt(x_rot**2 + (y_rot / axis_ratio[:, None, None]) ** 2)
        sigma = width[:, None, None] / 2.3548200450309493
        image = torch.exp(-0.5 * ((elliptical_radius - radius[:, None, None]) / sigma) ** 2)
    else:
        raise ValueError(f"torch MCMC example does not support component kind {component.kind!r}")

    flat = image.reshape(n_chains, -1)
    total = torch.sum(flat, dim=1, keepdim=True)
    return torch.where(total > 0, flat / total, flat)


def _power_law(flux, spectral_index, wavelengths, lambda_0):
    return flux[:, None] * (wavelengths[None, :] / lambda_0[:, None]) ** spectral_index[:, None]


def _model_cvis_torch(model, names, z, data_tensors, x_rad, y_rad, uv_chunk_size, dtype):
    torch = _import_torch()
    device = z.device
    n_chains = z.shape[0]
    name_to_index = {name: idx for idx, name in enumerate(names)}
    wavelengths = data_tensors["uv_lam"]
    n_uv = wavelengths.numel()
    cvis = torch.empty((n_chains, n_uv), device=device, dtype=torch.complex64)

    lambda_0 = _spec_value(model.sparco.lambda_0, "sparco.lambda_0", name_to_index, z, n_chains, device, dtype)
    star_flux_0 = _spec_value(model.sparco.star.flux, "sparco.star.flux", name_to_index, z, n_chains, device, dtype)
    star_index = _spec_value(
        model.sparco.star.spectral_index,
        "sparco.star.spectral_index",
        name_to_index,
        z,
        n_chains,
        device,
        dtype,
    )
    star_diameter = _spec_value(
        model.sparco.star.diameter,
        "sparco.star.diameter",
        name_to_index,
        z,
        n_chains,
        device,
        dtype,
    )
    if torch.any(torch.abs(star_diameter) > 0):
        raise ValueError("torch MCMC example currently supports unresolved stars only: set star.diameter to 0")

    component_images = [
        _render_component_torch(component, x_rad["mas"], y_rad["mas"], name_to_index, z, device, dtype)
        for component in model.components
    ]

    pix_x = x_rad["rad"].reshape(-1)
    pix_y = y_rad["rad"].reshape(-1)
    for start in range(0, n_uv, uv_chunk_size):
        stop = min(start + uv_chunk_size, n_uv)
        u = data_tensors["uv_u"][start:stop]
        v = data_tensors["uv_v"][start:stop]
        lam = wavelengths[start:stop]
        phase = -2.0 * np.pi * ((-u[:, None]) * pix_x[None, :] + v[:, None] * pix_y[None, :])
        kernel = torch.exp(1j * phase).to(torch.complex64)

        numerator = _power_law(star_flux_0, star_index, lam, lambda_0).to(torch.complex64)
        denominator = _power_law(star_flux_0, star_index, lam, lambda_0)

        halo = model.sparco.halo
        if halo is not None:
            halo_flux_0 = _spec_value(halo.flux, "sparco.halo.flux", name_to_index, z, n_chains, device, dtype)
            halo_index = _spec_value(
                halo.spectral_index,
                "sparco.halo.spectral_index",
                name_to_index,
                z,
                n_chains,
                device,
                dtype,
            )
            halo_visibility = _spec_value(
                halo.visibility,
                "sparco.halo.visibility",
                name_to_index,
                z,
                n_chains,
                device,
                dtype,
            )
            halo_flux = _power_law(halo_flux_0, halo_index, lam, lambda_0)
            numerator = numerator + (halo_flux * halo_visibility[:, None]).to(torch.complex64)
            denominator = denominator + halo_flux

        for component, image_flat in zip(model.components, component_images):
            comp_cvis = image_flat.to(torch.complex64) @ kernel.T
            comp_flux_0 = _spec_value(component.flux, f"{component.name}.flux", name_to_index, z, n_chains, device, dtype)
            comp_index = _spec_value(
                component.spectral_index,
                f"{component.name}.spectral_index",
                name_to_index,
                z,
                n_chains,
                device,
                dtype,
            )
            comp_flux = _power_law(comp_flux_0, comp_index, lam, lambda_0)
            numerator = numerator + comp_flux.to(torch.complex64) * comp_cvis
            denominator = denominator + comp_flux

        cvis[:, start:stop] = numerator / denominator.to(torch.complex64)

    return cvis


def _log_prob_batch(model, names, z, bounds, data_tensors, x_rad, y_rad, weights, uv_chunk_size, dtype):
    torch = _import_torch()
    valid = torch.isfinite(z).all(dim=1)
    for iparam, bound in enumerate(bounds):
        if bound is None:
            continue
        lower, upper = bound
        if lower is not None:
            valid &= z[:, iparam] >= float(lower)
        if upper is not None:
            valid &= z[:, iparam] <= float(upper)

    logp = torch.full((z.shape[0],), -torch.inf, device=z.device, dtype=dtype)
    if not torch.any(valid):
        return logp

    try:
        cvis = _model_cvis_torch(model, names, z[valid], data_tensors, x_rad, y_rad, uv_chunk_size, dtype)
        chi2 = torch.zeros(cvis.shape[0], device=z.device, dtype=dtype)
        if weights[0] > 0:
            v2 = torch.abs(cvis[:, data_tensors["indx_v2"]]) ** 2
            chi2 += float(weights[0]) * torch.sum(((v2 - data_tensors["v2"]) / data_tensors["v2_err"]) ** 2, dim=1)
        if weights[1] > 0:
            triple = (
                cvis[:, data_tensors["indx_t3_1"]]
                * cvis[:, data_tensors["indx_t3_2"]]
                * cvis[:, data_tensors["indx_t3_3"]]
            )
            t3amp = torch.abs(triple)
            chi2 += float(weights[1]) * torch.sum(((t3amp - data_tensors["t3amp"]) / data_tensors["t3amp_err"]) ** 2, dim=1)
        if weights[2] > 0:
            triple = (
                cvis[:, data_tensors["indx_t3_1"]]
                * cvis[:, data_tensors["indx_t3_2"]]
                * cvis[:, data_tensors["indx_t3_3"]]
            )
            t3phi = torch.angle(triple) * 180.0 / np.pi
            diff = torch.remainder(t3phi - data_tensors["t3phi"] + 180.0, 360.0) - 180.0
            chi2 += float(weights[2]) * torch.sum((diff / data_tensors["t3phi_err"]) ** 2, dim=1)
        logp[valid] = -0.5 * chi2
    except Exception:
        return logp
    return logp


def _initial_walkers(z0, bounds, n_walkers, scatter, seed, device, dtype):
    torch = _import_torch()
    rng = np.random.default_rng(seed)
    p0 = np.zeros((n_walkers, z0.size), dtype=float)
    for iparam, value in enumerate(z0):
        bound = bounds[iparam]
        if bound is not None and bound[0] is not None and bound[1] is not None:
            lower, upper = bound
            width = upper - lower
            p0[:, iparam] = value + scatter * width * rng.standard_normal(n_walkers)
            p0[:, iparam] = np.clip(p0[:, iparam], lower + 1e-10 * width, upper - 1e-10 * width)
        else:
            sigma = scatter * max(abs(value), 1.0)
            p0[:, iparam] = value + sigma * rng.standard_normal(n_walkers)
    return torch.as_tensor(p0, device=device, dtype=dtype)


def _proposal_scales(z0, bounds, proposal_scale, device, dtype):
    torch = _import_torch()
    scales = np.zeros(z0.size, dtype=float)
    for iparam, value in enumerate(z0):
        bound = bounds[iparam]
        if bound is not None and bound[0] is not None and bound[1] is not None:
            scales[iparam] = proposal_scale * (bound[1] - bound[0])
        else:
            scales[iparam] = proposal_scale * max(abs(value), 1.0)
    return torch.as_tensor(scales, device=device, dtype=dtype)


def _data_to_torch(data, device, dtype):
    torch = _import_torch()

    def tensor(value):
        return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)

    def index(value):
        return torch.as_tensor(np.asarray(value), device=device, dtype=torch.long)

    return {
        "uv_u": tensor(data.uv[0]),
        "uv_v": tensor(data.uv[1]),
        "uv_lam": tensor(data.uv_lam),
        "indx_v2": index(data.indx_v2),
        "indx_t3_1": index(data.indx_t3_1),
        "indx_t3_2": index(data.indx_t3_2),
        "indx_t3_3": index(data.indx_t3_3),
        "v2": tensor(data.v2),
        "v2_err": tensor(data.v2_err),
        "t3amp": tensor(data.t3amp),
        "t3amp_err": tensor(data.t3amp_err),
        "t3phi": tensor(data.t3phi),
        "t3phi_err": tensor(data.t3phi_err),
    }


def run_torch_mcmc(
    oifits_file: Path,
    spec_file: Path,
    output_dir: Path,
    nx: int = 32,
    pixsize_mas: float = 0.125,
    n_walkers: int | None = None,
    n_steps: int = 500,
    burn_in: int = 100,
    thin: int = 1,
    scatter: float = 0.01,
    proposal_scale: float = 0.01,
    seed: int = 42,
    device_name: str = "auto",
    uv_chunk_size: int = 2048,
    weights=(1.0, 0.0, 1.0),
):
    torch = _import_torch()
    try:
        from moifits.readoifits import readoifits
    except ModuleNotFoundError as exc:
        if exc.name == "astropy":
            raise SystemExit("Missing dependency 'astropy'. Install the package with `python -m pip install -e .`.") from exc
        raise

    device = _select_device(device_name)
    dtype = torch.float32
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_spec(spec_file)
    names = free_parameter_names(model)
    z0 = parameter_vector(model)
    bounds = parameter_bounds(model)
    if z0.size == 0:
        raise ValueError("model spec has no free parameters")
    if n_walkers is None:
        n_walkers = max(32, 2 * z0.size + 2)
    if burn_in >= n_steps:
        raise ValueError("burn_in must be smaller than n_steps")
    if thin < 1:
        raise ValueError("thin must be >= 1")

    data = readoifits(str(oifits_file), filter_bad_data=True, redundance_remove=True)
    data_tensors = _data_to_torch(data, device, dtype)
    x_mas, y_mas = make_coordinate_grid(nx, pixel_scale=pixsize_mas)
    mas_to_rad = np.pi / 180.0 / 3.6e6
    x_rad = {
        "mas": torch.as_tensor(x_mas, device=device, dtype=dtype),
        "rad": torch.as_tensor(x_mas * mas_to_rad, device=device, dtype=dtype),
    }
    y_rad = {
        "mas": torch.as_tensor(y_mas, device=device, dtype=dtype),
        "rad": torch.as_tensor(y_mas * mas_to_rad, device=device, dtype=dtype),
    }

    print(f"Device: {device}")
    print("Free parameters:")
    for name, value, bound in zip(names, z0, bounds):
        print(f"  {name:28s} start={value: .6g} bounds={bound}")

    current = _initial_walkers(z0, bounds, n_walkers, scatter, seed, device, dtype)
    scales = _proposal_scales(z0, bounds, proposal_scale, device, dtype)
    logp = _log_prob_batch(model, names, current, bounds, data_tensors, x_rad, y_rad, weights, uv_chunk_size, dtype)
    chain = torch.empty((n_steps, n_walkers, z0.size), device="cpu", dtype=dtype)
    log_prob_chain = torch.empty((n_steps, n_walkers), device="cpu", dtype=dtype)
    accepted = torch.zeros((n_walkers,), device=device, dtype=dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1)
    for step in range(n_steps):
        proposal = current + torch.randn(current.shape, device=device, dtype=dtype, generator=generator) * scales
        proposal_logp = _log_prob_batch(model, names, proposal, bounds, data_tensors, x_rad, y_rad, weights, uv_chunk_size, dtype)
        log_alpha = proposal_logp - logp
        accept = torch.log(torch.rand(n_walkers, device=device, dtype=dtype, generator=generator)) < log_alpha
        current = torch.where(accept[:, None], proposal, current)
        logp = torch.where(accept, proposal_logp, logp)
        accepted += accept.to(dtype)
        chain[step] = current.detach().cpu()
        log_prob_chain[step] = logp.detach().cpu()
        if (step + 1) % max(1, n_steps // 10) == 0:
            print(f"step {step + 1}/{n_steps} acceptance={float(torch.mean(accepted / (step + 1))):.3f}")

    posterior_chain = chain[burn_in::thin].numpy()
    posterior_logp = log_prob_chain[burn_in::thin].numpy()
    flat_chain = posterior_chain.reshape(-1, z0.size)
    flat_logp = posterior_logp.reshape(-1)
    finite = np.isfinite(flat_logp)
    if not np.any(finite):
        raise RuntimeError("sampler produced no finite posterior samples")
    best_idx = np.argmax(np.where(finite, flat_logp, -np.inf))
    best_parameters = flat_chain[best_idx]
    median_parameters = np.median(flat_chain[finite], axis=0)
    std_parameters = np.std(flat_chain[finite], axis=0)
    best_model = update_model_parameters(model, best_parameters, copy_model=True)

    best_spec_path = output_dir / "best_model_spec.json"
    with best_spec_path.open("w", encoding="utf-8") as stream:
        json.dump(model_spec_to_dict(best_model), stream, indent=2)
        stream.write("\n")

    image = render_model(best_model, x_mas, y_mas)
    np.save(output_dir / "best_model_image.npy", image)
    np.savez(
        output_dir / "torch_mcmc_result.npz",
        parameter_names=np.array(names),
        initial_parameters=z0,
        best_parameters=best_parameters,
        median_parameters=median_parameters,
        std_parameters=std_parameters,
        chain=chain.numpy(),
        log_prob=log_prob_chain.numpy(),
        acceptance_fraction=(accepted / n_steps).detach().cpu().numpy(),
        device=str(device),
    )

    try:
        import matplotlib.pyplot as plt

        extent = 0.5 * nx * pixsize_mas
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(image, origin="lower", extent=[-extent, extent, -extent, extent], cmap="viridis")
        ax.set_xlabel("RA offset (mas)")
        ax.set_ylabel("Dec offset (mas)")
        ax.set_title("Best-fit model image")
        fig.colorbar(im, ax=ax, label="Normalized intensity")
        fig.tight_layout()
        fig.savefig(output_dir / "best_model_image.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(z0.size, 1, figsize=(9, max(2, 1.5 * z0.size)), sharex=True)
        if z0.size == 1:
            axes = [axes]
        full_chain = chain.numpy()
        for iparam, ax in enumerate(axes):
            ax.plot(full_chain[:, :, iparam], color="black", alpha=0.25, linewidth=0.6)
            ax.set_ylabel(names[iparam], fontsize=8)
        axes[-1].set_xlabel("step")
        fig.tight_layout()
        fig.savefig(output_dir / "chains.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Skipping plots: {exc}")

    print("")
    print(f"mean acceptance fraction: {float(torch.mean(accepted / n_steps)):.3f}")
    print("Best parameters:")
    for name, value in zip(names, best_parameters):
        print(f"  {name:28s} {value: .8g}")
    print("Median parameters:")
    for name, value, std in zip(names, median_parameters, std_parameters):
        print(f"  {name:28s} {value: .8g} +/- {std:.3g}")
    print(f"Saved best spec: {best_spec_path}")
    return chain, best_model


def main():
    parser = argparse.ArgumentParser(description="Fit a JSON model spec with PyTorch MCMC.")
    parser.add_argument(
        "oifits_file",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("synthetic_from_image.oifits"),
        help="Input OIFITS file.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).with_name("fit_model_spec.json"),
        help="Input JSON model spec.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("torch_fit_output"), help="Directory for fit outputs.")
    parser.add_argument("--nx", type=int, default=32, help="Model image size.")
    parser.add_argument("--pixsize", type=float, default=0.125, help="Pixel scale in mas.")
    parser.add_argument("--n-walkers", type=int, default=None, help="Number of parallel walkers.")
    parser.add_argument("--n-steps", type=int, default=500, help="Number of MCMC steps.")
    parser.add_argument("--burn-in", type=int, default=100, help="Initial steps to discard.")
    parser.add_argument("--thin", type=int, default=1, help="Posterior thinning factor.")
    parser.add_argument("--scatter", type=float, default=0.01, help="Initial walker scatter as a fraction of bounds.")
    parser.add_argument("--proposal-scale", type=float, default=0.01, help="Proposal width as a fraction of bounds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto", help="Torch device.")
    parser.add_argument("--uv-chunk-size", type=int, default=2048, help="UV samples per DFT chunk.")
    parser.add_argument("--w-v2", type=float, default=1.0, help="VIS2 chi2 weight.")
    parser.add_argument("--w-t3amp", type=float, default=0.0, help="T3AMP chi2 weight.")
    parser.add_argument("--w-t3phi", type=float, default=1.0, help="T3PHI chi2 weight.")
    args = parser.parse_args()

    run_torch_mcmc(
        oifits_file=args.oifits_file,
        spec_file=args.spec,
        output_dir=args.output_dir,
        nx=args.nx,
        pixsize_mas=args.pixsize,
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
        thin=args.thin,
        scatter=args.scatter,
        proposal_scale=args.proposal_scale,
        seed=args.seed,
        device_name=args.device,
        uv_chunk_size=args.uv_chunk_size,
        weights=(args.w_v2, args.w_t3amp, args.w_t3phi),
    )


if __name__ == "__main__":
    main()
