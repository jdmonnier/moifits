import argparse
import json
from pathlib import Path

import numpy as np

from moifits.models import (
    ModelSpec,
    component_from_dict,
    load_model_spec,
    make_coordinate_grid,
    render_model,
    single_component_model,
)


def _json_arg(value: str) -> dict:
    if value.startswith("@"):
        with Path(value[1:]).open("r", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(value)


def _single_component_from_args(args: argparse.Namespace) -> ModelSpec:
    if args.kind in {"ring", "elliptical-ring"}:
        radius = args.radius if args.radius is not None else args.diameter / 2.0
        return single_component_model(
            args.kind,
            normalize=args.normalize,
            radius=radius,
            width=args.width,
            axis_ratio=args.axis_ratio,
            pa_deg=args.pa,
            x0=args.x0,
            y0=args.y0,
            flux=args.flux,
            spectral_index=args.spectral_index,
        )

    fwhm_major = args.fwhm_major if args.fwhm_major is not None else args.fwhm
    fwhm_minor = args.fwhm_minor if args.fwhm_minor is not None else fwhm_major
    return single_component_model(
        "gaussian",
        normalize=args.normalize,
        fwhm_major=fwhm_major,
        fwhm_minor=fwhm_minor,
        pa_deg=args.pa,
        x0=args.x0,
        y0=args.y0,
        flux=args.flux,
        spectral_index=args.spectral_index,
    )


def _model_from_args(args: argparse.Namespace) -> ModelSpec:
    if args.spec is not None:
        return load_model_spec(args.spec)
    if args.component:
        return ModelSpec(
            components=[component_from_dict(_json_arg(item)) for item in args.component],
            normalize=args.normalize,
        )
    return _single_component_from_args(args)


def main():
    parser = argparse.ArgumentParser(description="Generate parametric model images.")
    parser.add_argument("--npix", type=int, default=128, help="Image size (npix x npix).")
    parser.add_argument("--pixel-scale", type=float, default=1.0, help="Coordinate scale per pixel.")
    parser.add_argument("--output", type=str, default="model.npy", help="Output .npy filename.")
    parser.add_argument(
        "--kind",
        choices=["ring", "elliptical-ring", "gaussian"],
        default="ring",
        help="Single-component model kind used when --spec/--component are omitted.",
    )
    parser.add_argument("--diameter", type=float, default=64.0, help="Ring diameter.")
    parser.add_argument("--radius", type=float, default=None, help="Ring radius. Overrides --diameter.")
    parser.add_argument("--width", type=float, default=8.0, help="Ring radial FWHM.")
    parser.add_argument("--axis-ratio", type=float, default=1.0, help="Minor/major axis ratio for elliptical rings.")
    parser.add_argument("--fwhm", type=float, default=16.0, help="Circular Gaussian FWHM.")
    parser.add_argument("--fwhm-major", type=float, default=None, help="Gaussian major-axis FWHM.")
    parser.add_argument("--fwhm-minor", type=float, default=None, help="Gaussian minor-axis FWHM.")
    parser.add_argument("--pa", type=float, default=0.0, help="Position angle in degrees.")
    parser.add_argument("--x0", type=float, default=0.0, help="Component x offset in coordinate units.")
    parser.add_argument("--y0", type=float, default=0.0, help="Component y offset in coordinate units.")
    parser.add_argument("--flux", type=float, default=1.0, help="Single-component flux scale.")
    parser.add_argument("--spectral-index", type=float, default=0.0, help="Single-component SPARCO spectral index metadata.")
    parser.add_argument(
        "--component",
        action="append",
        help="JSON component object, or @path/to/component.json. Repeat for multiple components.",
    )
    parser.add_argument("--spec", type=Path, default=None, help="Full JSON model spec.")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize", help="Do not normalize total flux.")
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    model = _model_from_args(args)
    x_grid, y_grid = make_coordinate_grid(args.npix, pixel_scale=args.pixel_scale)
    image = render_model(model, x_grid, y_grid)

    output = Path(args.output)
    np.save(output, image)
    png_path = output.with_suffix(".png")
    import matplotlib.pyplot as plt

    plt.imsave(png_path, image, cmap="gray")
    component_names = ", ".join(component.name for component in model.components)
    print(f"Saved model image to {output} and {png_path} with shape {image.shape}")
    print(f"Components: {component_names}")


if __name__ == "__main__":
    main()
