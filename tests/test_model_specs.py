import numpy as np

from moifits.models import (
    free_parameter_names,
    model_spec_from_dict,
    model_spec_to_dict,
    parameter_bounds,
    parameter_vector,
    update_model_parameters,
)
from moifits.sparco import power_law_flux


def _example_spec():
    return {
        "normalize": True,
        "sparco": {
            "lambda_0": 1.65e-6,
            "star": {
                "flux": {"value": 0.4, "fixed": False, "bounds": [0.0, 1.0]},
                "diameter": {"value": 0.0, "fixed": True},
                "spectral_index": {"value": -4.0, "fixed": False, "bounds": [-6.0, 2.0]},
            },
            "halo": {
                "flux": {"value": 0.1, "fixed": False, "bounds": [0.0, 1.0]},
                "spectral_index": 0.0,
                "visibility": 1.0,
            },
        },
        "components": [
            {
                "name": "ring",
                "kind": "elliptical-ring",
                "flux": {"value": 0.5, "fixed": False, "bounds": [0.0, 1.0]},
                "spectral_index": {"value": 0.0, "fixed": False, "bounds": [-5.0, 5.0]},
                "params": {
                    "radius": {"value": 2.0, "fixed": False, "bounds": [0.1, 10.0]},
                    "width": 0.5,
                    "axis_ratio": {"value": 0.7, "fixed": True},
                },
            }
        ],
    }


def test_model_spec_tracks_sparco_and_component_free_parameters():
    model = model_spec_from_dict(_example_spec())

    assert free_parameter_names(model) == [
        "sparco.star.flux",
        "sparco.star.spectral_index",
        "sparco.halo.flux",
        "ring.flux",
        "ring.spectral_index",
        "ring.radius",
    ]
    assert parameter_vector(model).tolist() == [0.4, -4.0, 0.1, 0.5, 0.0, 2.0]
    assert parameter_bounds(model)[0] == (0.0, 1.0)

    updated = update_model_parameters(model, np.array([0.3, -3.5, 0.2, 0.4, 1.0, 3.0]))
    assert updated.sparco.star.flux.value == 0.3
    assert updated.sparco.star.spectral_index.value == -3.5
    assert updated.sparco.halo.flux.value == 0.2
    assert updated.components[0].flux.value == 0.4
    assert updated.components[0].spectral_index.value == 1.0
    assert updated.components[0].params["radius"].value == 3.0


def test_model_spec_round_trips_to_json_compatible_dict():
    model = model_spec_from_dict(_example_spec())
    as_dict = model_spec_to_dict(model)

    assert as_dict["sparco"]["star"]["spectral_index"]["value"] == -4.0
    assert as_dict["sparco"]["halo"]["visibility"]["value"] == 1.0
    assert as_dict["components"][0]["spectral_index"]["fixed"] is False


def test_power_law_flux_uses_explicit_spectral_index():
    wavelengths = np.array([1.65e-6, 3.30e-6])
    flux = power_law_flux(2.0, -1.0, wavelengths, 1.65e-6)

    np.testing.assert_allclose(flux, [2.0, 1.0])


if __name__ == "__main__":
    test_model_spec_tracks_sparco_and_component_free_parameters()
    test_model_spec_round_trips_to_json_compatible_dict()
    test_power_law_flux_uses_explicit_spectral_index()
