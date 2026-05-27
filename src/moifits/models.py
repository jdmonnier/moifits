"""Parametric image components for OI model generation and fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class ParameterSpec:
    """A model parameter value plus fitting metadata."""

    value: float
    fixed: bool = False
    bounds: Optional[Tuple[float, float]] = None


@dataclass
class StarSpec:
    """Model-level stellar SPARCO contribution."""

    flux: ParameterSpec = field(default_factory=lambda: ParameterSpec(0.0, fixed=True))
    diameter: ParameterSpec = field(default_factory=lambda: ParameterSpec(0.0, fixed=True))
    spectral_index: ParameterSpec = field(default_factory=lambda: ParameterSpec(-4.0, fixed=True))


@dataclass
class HaloSpec:
    """Model-level unresolved or partially resolved halo contribution."""

    flux: ParameterSpec = field(default_factory=lambda: ParameterSpec(0.0, fixed=True))
    spectral_index: ParameterSpec = field(default_factory=lambda: ParameterSpec(0.0, fixed=True))
    visibility: ParameterSpec = field(default_factory=lambda: ParameterSpec(1.0, fixed=True))


@dataclass
class SparcoSpec:
    """Model-level SPARCO flux law configuration."""

    lambda_0: ParameterSpec = field(default_factory=lambda: ParameterSpec(1.65e-6, fixed=True))
    star: StarSpec = field(default_factory=StarSpec)
    halo: Optional[HaloSpec] = None


@dataclass
class ComponentSpec:
    """A renderable model component."""

    kind: str
    name: str = "component"
    flux: ParameterSpec = field(default_factory=lambda: ParameterSpec(1.0, fixed=True))
    spectral_index: ParameterSpec = field(default_factory=lambda: ParameterSpec(0.0, fixed=True))
    params: Dict[str, ParameterSpec] = field(default_factory=dict)


@dataclass
class ModelSpec:
    """A full image model made from one or more components."""

    components: list[ComponentSpec]
    normalize: bool = True
    sparco: SparcoSpec = field(default_factory=SparcoSpec)


def make_coordinate_grid(npix: int, pixel_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return centered image coordinates in pixel_scale units."""
    coords = (np.arange(npix, dtype=float) - (npix - 1) / 2.0) * float(pixel_scale)
    return np.meshgrid(coords, coords, indexing="ij")


def _parameter_from_dict(value: Any) -> ParameterSpec:
    if isinstance(value, ParameterSpec):
        return value
    if isinstance(value, dict):
        bounds = value.get("bounds")
        return ParameterSpec(
            value=float(value["value"]),
            fixed=bool(value.get("fixed", False)),
            bounds=tuple(bounds) if bounds is not None else None,
        )
    return ParameterSpec(float(value), fixed=True)


def _parameter_to_dict(param: ParameterSpec) -> Dict[str, Any]:
    result: Dict[str, Any] = {"value": param.value, "fixed": param.fixed}
    if param.bounds is not None:
        result["bounds"] = list(param.bounds)
    return result


def _star_from_dict(data: Dict[str, Any]) -> StarSpec:
    return StarSpec(
        flux=_parameter_from_dict(data.get("flux", 0.0)),
        diameter=_parameter_from_dict(data.get("diameter", 0.0)),
        spectral_index=_parameter_from_dict(data.get("spectral_index", -4.0)),
    )


def _halo_from_dict(data: Optional[Dict[str, Any]]) -> Optional[HaloSpec]:
    if data is None:
        return None
    return HaloSpec(
        flux=_parameter_from_dict(data.get("flux", 0.0)),
        spectral_index=_parameter_from_dict(data.get("spectral_index", 0.0)),
        visibility=_parameter_from_dict(data.get("visibility", 1.0)),
    )


def _sparco_from_dict(data: Optional[Dict[str, Any]]) -> SparcoSpec:
    if data is None:
        return SparcoSpec()
    return SparcoSpec(
        lambda_0=_parameter_from_dict(data.get("lambda_0", 1.65e-6)),
        star=_star_from_dict(data.get("star", {})),
        halo=_halo_from_dict(data.get("halo")),
    )


def component_from_dict(data: Dict[str, Any]) -> ComponentSpec:
    """Build a ComponentSpec from a JSON-compatible dictionary."""
    params = {
        key: _parameter_from_dict(value)
        for key, value in data.get("params", {}).items()
    }
    flux = _parameter_from_dict(data.get("flux", 1.0))
    return ComponentSpec(
        kind=str(data["kind"]),
        name=str(data.get("name", data["kind"])),
        flux=flux,
        spectral_index=_parameter_from_dict(data.get("spectral_index", 0.0)),
        params=params,
    )


def model_spec_from_dict(data: Dict[str, Any]) -> ModelSpec:
    """Build a ModelSpec from a JSON-compatible dictionary."""
    return ModelSpec(
        components=[component_from_dict(component) for component in data["components"]],
        normalize=bool(data.get("normalize", True)),
        sparco=_sparco_from_dict(data.get("sparco")),
    )


def load_model_spec(path: str | Path) -> ModelSpec:
    """Load a ModelSpec from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return model_spec_from_dict(json.load(stream))


def component_to_dict(component: ComponentSpec) -> Dict[str, Any]:
    """Convert a component to a JSON-compatible dictionary."""
    return {
        "name": component.name,
        "kind": component.kind,
        "flux": _parameter_to_dict(component.flux),
        "spectral_index": _parameter_to_dict(component.spectral_index),
        "params": {
            name: _parameter_to_dict(param)
            for name, param in component.params.items()
        },
    }


def _star_to_dict(star: StarSpec) -> Dict[str, Any]:
    return {
        "flux": _parameter_to_dict(star.flux),
        "diameter": _parameter_to_dict(star.diameter),
        "spectral_index": _parameter_to_dict(star.spectral_index),
    }


def _halo_to_dict(halo: Optional[HaloSpec]) -> Optional[Dict[str, Any]]:
    if halo is None:
        return None
    return {
        "flux": _parameter_to_dict(halo.flux),
        "spectral_index": _parameter_to_dict(halo.spectral_index),
        "visibility": _parameter_to_dict(halo.visibility),
    }


def _sparco_to_dict(sparco: SparcoSpec) -> Dict[str, Any]:
    result = {
        "lambda_0": _parameter_to_dict(sparco.lambda_0),
        "star": _star_to_dict(sparco.star),
    }
    if sparco.halo is not None:
        result["halo"] = _halo_to_dict(sparco.halo)
    return result


def model_spec_to_dict(model: ModelSpec) -> Dict[str, Any]:
    """Convert a ModelSpec to a JSON-compatible dictionary."""
    return {
        "normalize": model.normalize,
        "sparco": _sparco_to_dict(model.sparco),
        "components": [component_to_dict(component) for component in model.components],
    }


def free_parameters(model: ModelSpec) -> list[tuple[str, str, ParameterSpec]]:
    """Return model and component parameters marked free for fitting."""
    free: list[tuple[str, str, ParameterSpec]] = []
    for owner, name, param in iter_parameter_specs(model):
        if not param.fixed:
            free.append((owner, name, param))
    return free


def iter_parameter_specs(model: ModelSpec):
    """Yield all ParameterSpec objects as (owner, name, spec)."""
    yield "sparco", "lambda_0", model.sparco.lambda_0
    yield "sparco.star", "flux", model.sparco.star.flux
    yield "sparco.star", "diameter", model.sparco.star.diameter
    yield "sparco.star", "spectral_index", model.sparco.star.spectral_index
    if model.sparco.halo is not None:
        yield "sparco.halo", "flux", model.sparco.halo.flux
        yield "sparco.halo", "spectral_index", model.sparco.halo.spectral_index
        yield "sparco.halo", "visibility", model.sparco.halo.visibility
    for component in model.components:
        yield component.name, "flux", component.flux
        yield component.name, "spectral_index", component.spectral_index
        for name, param in component.params.items():
            yield component.name, name, param


def free_parameter_names(model: ModelSpec) -> list[str]:
    """Return stable names for free parameters in model-vector order."""
    return [f"{owner}.{name}" for owner, name, _ in free_parameters(model)]


def parameter_vector(model: ModelSpec) -> np.ndarray:
    """Return the free-parameter initial vector."""
    return np.array([param.value for _, _, param in free_parameters(model)], dtype=float)


def parameter_bounds(model: ModelSpec) -> list[Optional[Tuple[float, float]]]:
    """Return free-parameter bounds in vector order."""
    return [param.bounds for _, _, param in free_parameters(model)]


def update_model_parameters(model: ModelSpec, values: np.ndarray, copy_model: bool = True) -> ModelSpec:
    """Write a free-parameter vector back into a model spec."""
    target = copy.deepcopy(model) if copy_model else model
    free = free_parameters(target)
    if len(values) != len(free):
        raise ValueError(f"expected {len(free)} values, got {len(values)}")
    for value, (_, _, param) in zip(values, free):
        param.value = float(value)
    return target


def _value(params: Dict[str, ParameterSpec], name: str, default: float) -> float:
    param = params.get(name)
    return float(param.value) if param is not None else float(default)


def _rotated_offsets(
    x: np.ndarray,
    y: np.ndarray,
    x0: float = 0.0,
    y0: float = 0.0,
    pa_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(pa_deg)
    dx = x - x0
    dy = y - y0
    x_rot = dx * np.cos(theta) + dy * np.sin(theta)
    y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
    return x_rot, y_rot


def render_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    fwhm_major: float,
    fwhm_minor: Optional[float] = None,
    pa_deg: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    """Render an elliptical Gaussian component."""
    if fwhm_major <= 0:
        raise ValueError("fwhm_major must be > 0")
    if fwhm_minor is None:
        fwhm_minor = fwhm_major
    if fwhm_minor <= 0:
        raise ValueError("fwhm_minor must be > 0")
    sigma_major = fwhm_major / 2.3548200450309493
    sigma_minor = fwhm_minor / 2.3548200450309493
    x_rot, y_rot = _rotated_offsets(x, y, x0=x0, y0=y0, pa_deg=pa_deg)
    exponent = -0.5 * ((x_rot / sigma_major) ** 2 + (y_rot / sigma_minor) ** 2)
    return np.exp(exponent)


def render_ring(
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    width: float,
    axis_ratio: float = 1.0,
    pa_deg: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    """Render a Gaussian radial ring, circular or elliptical."""
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if width <= 0:
        raise ValueError("width must be > 0")
    if not (0 < axis_ratio <= 1):
        raise ValueError("axis_ratio must be in (0, 1]")
    x_rot, y_rot = _rotated_offsets(x, y, x0=x0, y0=y0, pa_deg=pa_deg)
    elliptical_radius = np.sqrt(x_rot**2 + (y_rot / axis_ratio) ** 2)
    sigma = width / 2.3548200450309493
    return np.exp(-0.5 * ((elliptical_radius - radius) / sigma) ** 2)


def render_smooth_disk_ring(
    x: np.ndarray,
    y: np.ndarray,
    rin: float,
    rout: float,
    contrast: float,
    sigma_in: float,
    sigma_out: float,
    inclination_rad: float = 0.0,
    pa_rad: float = 0.0,
    apply_cutoff: bool = True,
) -> np.ndarray:
    """Render the smooth disk-ring model used by the MCMC example."""
    cos_i = np.cos(inclination_rad)
    r = np.hypot(x, y)
    theta_ang = np.arctan2(y, x)
    dtheta = theta_ang - pa_rad
    denom = np.sqrt(cos_i**2 * np.cos(dtheta) ** 2 + np.sin(dtheta) ** 2)
    denom = np.where(denom == 0, 1e-12, denom)
    projected_radius = r / (cos_i / denom)

    arg_inner = np.clip(-(projected_radius - rin) / sigma_in, -50, 50)
    arg_outer = np.clip((projected_radius - rout) / sigma_out, -50, 50)
    inner = contrast + (1.0 - contrast) / (1.0 + np.exp(arg_inner))
    outer = 1.0 / (1.0 + np.exp(arg_outer))
    image = inner * outer
    if apply_cutoff:
        image = np.where(image < np.max(image) / 100.0, 0.0, image)
    return image


def render_component(
    component: ComponentSpec,
    x: np.ndarray,
    y: np.ndarray,
    include_flux: bool = True,
) -> np.ndarray:
    """Render one component from a ComponentSpec."""
    params = component.params
    kind = component.kind.replace("_", "-").lower()

    if kind == "gaussian":
        image = render_gaussian(
            x,
            y,
            fwhm_major=_value(params, "fwhm_major", _value(params, "fwhm", 10.0)),
            fwhm_minor=_value(params, "fwhm_minor", _value(params, "fwhm", 10.0)),
            pa_deg=_value(params, "pa_deg", _value(params, "pa", 0.0)),
            x0=_value(params, "x0", 0.0),
            y0=_value(params, "y0", 0.0),
        )
    elif kind in {"ring", "elliptical-ring"}:
        image = render_ring(
            x,
            y,
            radius=_value(params, "radius", 32.0),
            width=_value(params, "width", 8.0),
            axis_ratio=_value(params, "axis_ratio", 1.0),
            pa_deg=_value(params, "pa_deg", _value(params, "pa", 0.0)),
            x0=_value(params, "x0", 0.0),
            y0=_value(params, "y0", 0.0),
        )
    elif kind in {"smooth-disk-ring", "disk-ring"}:
        image = render_smooth_disk_ring(
            x,
            y,
            rin=_value(params, "rin", 1.0),
            rout=_value(params, "rout", 2.0),
            contrast=_value(params, "contrast", 0.9),
            sigma_in=_value(params, "sigma_in", _value(params, "sigma", 0.02)),
            sigma_out=_value(params, "sigma_out", _value(params, "sigma", 0.02)),
            inclination_rad=_value(params, "inclination_rad", _value(params, "inc", 0.0)),
            pa_rad=_value(params, "pa_rad", _value(params, "phi", 0.0)),
            apply_cutoff=bool(_value(params, "apply_cutoff", 1.0)),
        )
    else:
        raise ValueError(f"Unsupported component kind: {component.kind}")

    if include_flux:
        image = float(component.flux.value) * image
    return image


def render_model(
    model: ModelSpec,
    x: np.ndarray,
    y: np.ndarray,
    normalize: Optional[bool] = None,
) -> np.ndarray:
    """Render all components in a model spec."""
    image = np.zeros_like(x, dtype=float)
    for component in model.components:
        image += render_component(component, x, y, include_flux=True)

    should_normalize = model.normalize if normalize is None else normalize
    if should_normalize:
        total = np.sum(image)
        if total > 0:
            image = image / total
    return image


def single_component_model(
    kind: str,
    normalize: bool = True,
    **params: float,
) -> ModelSpec:
    """Convenience constructor for one-component generated images."""
    flux = params.pop("flux", 1.0)
    spectral_index = params.pop("spectral_index", 0.0)
    return ModelSpec(
        components=[
            ComponentSpec(
                kind=kind,
                name=kind,
                flux=ParameterSpec(float(flux), fixed=True),
                spectral_index=ParameterSpec(float(spectral_index), fixed=True),
                params={key: ParameterSpec(float(value), fixed=True) for key, value in params.items()},
            )
        ],
        normalize=normalize,
    )


__all__ = [
    "ComponentSpec",
    "HaloSpec",
    "ModelSpec",
    "ParameterSpec",
    "SparcoSpec",
    "StarSpec",
    "component_from_dict",
    "component_to_dict",
    "free_parameters",
    "free_parameter_names",
    "iter_parameter_specs",
    "load_model_spec",
    "make_coordinate_grid",
    "model_spec_from_dict",
    "model_spec_to_dict",
    "parameter_bounds",
    "parameter_vector",
    "render_component",
    "render_gaussian",
    "render_model",
    "render_ring",
    "render_smooth_disk_ring",
    "single_component_model",
    "update_model_parameters",
]
