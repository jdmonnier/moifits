# moifits
Michigan OI-FITS Tools

## Layout

```text
src/moifits/        importable library code
src/moifits/cli/    command-line entry points
tests/              regression and comparison tests
examples/           runnable research examples and sample artifacts
```

Install in editable mode from the repository root:

```bash
python -m pip install -e .
```

Available command-line runners:

```bash
moifits-show file.oifits
moifits-plot file.oifits --save overview.png
moifits-generate-model --kind elliptical-ring --diameter 64 --width 8 --axis-ratio 0.6 --pa 35 --output ring.npy
moifits-generate-model --kind gaussian --fwhm-major 20 --fwhm-minor 8 --pa 45 --output gaussian.npy
moifits-image-to-oifits --image ring.npy --output synthetic_from_image.oifits
moifits-compare-image data.oifits ring.npy --pixsize 0.125 --f-star 0.4 --f-halo 0.1 --star-index -4 --halo-index 0 --halo-visibility 0 --image-index 0 --save compare.png
```

## Compare Image vs OIFITS

`moifits-compare-image` compares an input image model against OIFITS observables
with a SPARCO-style flux split (star + resolved halo + image), wavelength power
laws, and model/data comparison plots.

Default behavior:
- plots model curves at 5 equally spaced wavelengths
- uses dense uv sampling for plotted `V2` model lines
- computes residuals at the original OIFITS uv samples
- plots `T3PHI` model as points

Example (default wavelength selection):

```bash
moifits-compare-image data.oifits ring.npy \
  --pixsize 0.125 \
  --f-star 0.4 \
  --f-halo 0.1 \
  --star-index -4 \
  --halo-index 0 \
  --halo-visibility 0 \
  --image-index 0 \
  --save compare.png
```

Example (explicit wavelengths in microns):

```bash
moifits-compare-image data.oifits ring.npy \
  --pixsize 0.125 \
  --plot-wavelengths 1.53 1.60 1.67 1.74 1.81 \
  --save compare_selected_lam.png
```

## Model Specs

Parametric images are represented as component specs so generation and fitting
can eventually share the same inputs. A component parameter can be a bare value,
or an object with `value`, `fixed`, and optional `bounds`.

```json
{
  "normalize": true,
  "sparco": {
    "lambda_0": 1.65e-6,
    "star": {
      "flux": {"value": 0.4, "fixed": false, "bounds": [0.0, 1.0]},
      "diameter": {"value": 0.0, "fixed": true},
      "spectral_index": {"value": -4.0, "fixed": false, "bounds": [-6.0, 2.0]}
    },
    "halo": {
      "flux": {"value": 0.0, "fixed": true},
      "spectral_index": {"value": 0.0, "fixed": true},
      "visibility": {"value": 1.0, "fixed": true}
    }
  },
  "components": [
    {
      "name": "ring_1",
      "kind": "elliptical-ring",
      "flux": {"value": 0.8, "fixed": false, "bounds": [0.0, 1.0]},
      "spectral_index": {"value": 0.0, "fixed": false, "bounds": [-6.0, 6.0]},
      "params": {
        "radius": {"value": 32.0, "fixed": false, "bounds": [1.0, 80.0]},
        "width": {"value": 8.0, "fixed": false, "bounds": [0.5, 30.0]},
        "axis_ratio": {"value": 0.6, "fixed": false, "bounds": [0.1, 1.0]},
        "pa_deg": {"value": 35.0, "fixed": false, "bounds": [0.0, 180.0]},
        "x0": 0.0,
        "y0": 0.0
      }
    },
    {
      "name": "halo",
      "kind": "gaussian",
      "flux": {"value": 0.2, "fixed": false, "bounds": [0.0, 1.0]},
      "spectral_index": {"value": 0.0, "fixed": false, "bounds": [-6.0, 6.0]},
      "params": {
        "fwhm_major": {"value": 40.0, "fixed": false, "bounds": [1.0, 100.0]},
        "fwhm_minor": {"value": 40.0, "fixed": true},
        "pa_deg": 0.0
      }
    }
  ]
}
```

Generate from a full spec:

```bash
moifits-generate-model --spec model.json --output model.npy
```

The `sparco` block is model-level. The star and optional halo are not image
components; they are combined with the environmental component visibilities at
each wavelength. `halo.visibility` defaults to `1.0` for an unresolved halo; use
`0.0` if you want fully resolved incoherent flux.

For fitting, `moifits.models.parameter_vector(model)`,
`moifits.models.parameter_bounds(model)`, and
`moifits.fitting.log_posterior_model_vector(...)` provide the bridge from this
spec to an optimizer or sampler.

Run the spec-driven fitting example:

```bash
python examples/fit_model_spec.py examples/synthetic_from_image.oifits \
  --spec examples/fit_model_spec.json \
  --output-dir examples/fit_output \
  --backend finufft \
  --n-steps 500
```

The example uses `emcee` with the free parameters and bounds declared in the
JSON spec, then writes `best_model_spec.json`, `best_model_image.npy`,
`best_model_image.png`, `chains.png`, and `fit_result.npz`.

The emcee example can use the CPU, GPU, or debug Fourier backend:

```bash
python examples/fit_model_spec.py examples/synthetic_from_image.oifits \
  --spec examples/fit_model_spec.json \
  --backend cufinufft \
  --gpu-device-id 0 \
  --output-dir examples/fit_output_gpu
```

`backend=finufft` uses the CPU FINUFFT package, `backend=cufinufft` uses the
CUDA cuFINUFFT package, and `backend=direct` uses a slow direct DFT useful for
small debugging cases. For GPU support, install `cufinufft` plus a
CUDA-compatible CuPy package for your CUDA version.

There is also a GPU-capable PyTorch MCMC example:

```bash
python examples/fit_model_spec_torch_mcmc.py examples/synthetic_from_image.oifits \
  --spec examples/fit_model_spec.json \
  --output-dir examples/torch_fit_output \
  --device auto \
  --nx 32 \
  --n-steps 500
```

This path uses a chunked direct DFT likelihood in torch, so it is meant for
small-image GPU experiments. The faster FINUFFT path is still the better default
for larger production fits. The torch example currently assumes the model-level
star is unresolved, so keep `sparco.star.diameter` fixed at `0.0`. We're currently looking at the best way to support NFFT likelihoods on a GPU, which would allow for us to take full advantage of the GPU for larger images and more complex models.
