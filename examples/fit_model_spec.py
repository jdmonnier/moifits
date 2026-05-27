#!/usr/bin/env python3
"""
Fit a JSON model spec to an OIFITS file with emcee.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running this example directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moifits.fitting import run_emcee_model_fit


def main():
    parser = argparse.ArgumentParser(description="Fit a JSON model spec to OIFITS data with emcee.")
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
    parser.add_argument("--output-dir", type=Path, default=Path("fit_output"), help="Directory for fit outputs.")
    parser.add_argument("--nx", type=int, default=128, help="Model image size.")
    parser.add_argument("--pixsize", type=float, default=0.125, help="Pixel scale in mas.")
    parser.add_argument(
        "--backend",
        choices=["finufft", "cufinufft", "direct"],
        default="finufft",
        help="Fourier backend.",
    )
    parser.add_argument("--gpu-device-id", type=int, default=None, help="CUDA GPU ID for cufinufft.")
    parser.add_argument("--n-walkers", type=int, default=None, help="Number of emcee walkers.")
    parser.add_argument("--n-steps", type=int, default=500, help="Number of emcee steps.")
    parser.add_argument("--burn-in", type=int, default=100, help="Initial steps to discard.")
    parser.add_argument("--thin", type=int, default=1, help="Posterior thinning factor.")
    parser.add_argument("--scatter", type=float, default=0.01, help="Initial walker scatter as a fraction of bounds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for walker initialization.")
    parser.add_argument("--w-v2", type=float, default=1.0, help="VIS2 chi2 weight.")
    parser.add_argument("--w-t3amp", type=float, default=0.0, help="T3AMP chi2 weight.")
    parser.add_argument("--w-t3phi", type=float, default=1.0, help="T3PHI chi2 weight.")
    args = parser.parse_args()

    backend_kwargs = {}
    if args.gpu_device_id is not None:
        backend_kwargs["gpu_device_id"] = args.gpu_device_id

    try:
        result = run_emcee_model_fit(
            oifits_file=args.oifits_file,
            spec_file=args.spec,
            output_dir=args.output_dir,
            nx=args.nx,
            pixsize_mas=args.pixsize,
            backend=args.backend,
            n_walkers=args.n_walkers,
            n_steps=args.n_steps,
            burn_in=args.burn_in,
            thin=args.thin,
            scatter=args.scatter,
            seed=args.seed,
            weights=(args.w_v2, args.w_t3amp, args.w_t3phi),
            **backend_kwargs,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print("")
    print(f"mean acceptance fraction: {result['acceptance_fraction'].mean():.3f}")
    print(f"chi2: {result['chi2']:.6g}")
    print(f"reduced chi2: {result['reduced_chi2']:.6g}")
    print("Best parameters:")
    for name, value in zip(result["parameter_names"], result["best_parameters"]):
        print(f"  {name:28s} {value: .8g}")
    print("Median parameters:")
    for name, value, std in zip(
        result["parameter_names"],
        result["median_parameters"],
        result["std_parameters"],
    ):
        print(f"  {name:28s} {value: .8g} +/- {std:.3g}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
