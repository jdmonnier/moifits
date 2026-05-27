"""
Utilities to create synthetic OIFITS files from a forward visibility model.

1) Build UV sampling from array geometry + hour angles.
2) Evaluate a user-provided complex visibility model on that sampling.
3) Write OIFITS tables (OI_TARGET, OI_WAVELENGTH, OI_VIS, OI_VIS2, OI_T3).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from astropy.io import fits


ComplexVisibilityFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]
# Backward-compatible alias kept for existing imports/usages.
ComplexVisibilityModel = ComplexVisibilityFn


@dataclass
class NoiseConfig:
    """Error bars used for synthetic observables and optional Gaussian noise."""

    visamp_err: float = 0.02
    visphi_err_deg: float = 2.0
    v2_err: float = 0.03
    t3amp_err: float = 0.05
    t3phi_err_deg: float = 3.0
    add_noise: bool = False
    seed: Optional[int] = None


def project_baseline_to_uv(
    baseline_enu_m: np.ndarray,
    hour_angle_rad: np.ndarray,
    dec_rad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project ENU baseline vectors into (u, v) in meters for a given declination.

    Args:
        baseline_enu_m: (3,) baseline vector [East, North, Up] in meters.
        hour_angle_rad: (ntimes,) hour angles in radians.
        dec_rad: Source declination in radians.

    Returns:
        u_m, v_m arrays of shape (ntimes,).
    """
    bx, by, bz = baseline_enu_m
    sin_h = np.sin(hour_angle_rad)
    cos_h = np.cos(hour_angle_rad)
    sin_d = np.sin(dec_rad)
    cos_d = np.cos(dec_rad)

    # Standard ENU -> UV projection.
    u_m = bx * sin_h + by * cos_h
    v_m = -bx * sin_d * cos_h + by * sin_d * sin_h + bz * cos_d
    return u_m, v_m


def generate_uv_sampling(
    station_enu_m: np.ndarray,
    hour_angles_rad: np.ndarray,
    dec_rad: float,
    mjd_times: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Generate VIS/V2/T3 sampling metadata from station geometry and hour angles.

    Args:
        station_enu_m: (nstations, 3) station coordinates in ENU meters.
        hour_angles_rad: (ntimes,) hour angles in radians.
        dec_rad: Source declination in radians.
        mjd_times: (ntimes,) MJD timestamps.

    Returns:
        Dict with VIS/V2 rows and T3 rows in OIFITS-friendly row format.
    """
    station_enu_m = np.asarray(station_enu_m, dtype=float)
    hour_angles_rad = np.asarray(hour_angles_rad, dtype=float)
    mjd_times = np.asarray(mjd_times, dtype=float)
    if station_enu_m.ndim != 2 or station_enu_m.shape[1] != 3:
        raise ValueError("station_enu_m must have shape (nstations, 3)")
    if hour_angles_rad.shape != mjd_times.shape:
        raise ValueError("hour_angles_rad and mjd_times must have the same shape")

    nstations = station_enu_m.shape[0]
    if nstations < 2:
        raise ValueError("At least 2 stations are required")

    pairs = list(combinations(range(nstations), 2))
    triangles = list(combinations(range(nstations), 3))

    vis_u: List[float] = []
    vis_v: List[float] = []
    vis_mjd: List[float] = []
    vis_sta: List[Tuple[int, int]] = []
    baseline_by_pair_time: Dict[Tuple[int, int, int], Tuple[float, float]] = {}

    for it, (ha, mjd) in enumerate(zip(hour_angles_rad, mjd_times)):
        for i, j in pairs:
            bl = station_enu_m[j] - station_enu_m[i]
            u_m, v_m = project_baseline_to_uv(bl, np.array([ha]), dec_rad)
            vis_u.append(u_m[0])
            vis_v.append(v_m[0])
            vis_mjd.append(mjd)
            vis_sta.append((i + 1, j + 1))  # OIFITS is 1-based station index
            baseline_by_pair_time[(it, i, j)] = (u_m[0], v_m[0])

    t3_u1: List[float] = []
    t3_v1: List[float] = []
    t3_u2: List[float] = []
    t3_v2: List[float] = []
    t3_mjd: List[float] = []
    t3_sta: List[Tuple[int, int, int]] = []

    # Keep T3 orientation consistent with readoifits.py expectations:
    # baseline1=(i,j), baseline2=(j,k), baseline3 inferred as baseline1+baseline2.
    for it, mjd in enumerate(mjd_times):
        for i, j, k in triangles:
            u1, v1 = baseline_by_pair_time[(it, i, j)]
            u2, v2 = baseline_by_pair_time[(it, j, k)]
            t3_u1.append(u1)
            t3_v1.append(v1)
            t3_u2.append(u2)
            t3_v2.append(v2)
            t3_mjd.append(mjd)
            t3_sta.append((i + 1, j + 1, k + 1))

    return {
        "vis_ucoord": np.asarray(vis_u, dtype=float),
        "vis_vcoord": np.asarray(vis_v, dtype=float),
        "vis_mjd": np.asarray(vis_mjd, dtype=float),
        "vis_sta_index": np.asarray(vis_sta, dtype=np.int16),
        "t3_u1coord": np.asarray(t3_u1, dtype=float),
        "t3_v1coord": np.asarray(t3_v1, dtype=float),
        "t3_u2coord": np.asarray(t3_u2, dtype=float),
        "t3_v2coord": np.asarray(t3_v2, dtype=float),
        "t3_mjd": np.asarray(t3_mjd, dtype=float),
        "t3_sta_index": np.asarray(t3_sta, dtype=np.int16),
    }


def _evaluate_cvis_grid(
    cvis_fn: ComplexVisibilityFn,
    ucoord_m: np.ndarray,
    vcoord_m: np.ndarray,
    wavelengths_m: np.ndarray,
) -> np.ndarray:
    """Evaluate model complex visibilities for all rows and spectral channels."""
    nrows = ucoord_m.size
    nwave = wavelengths_m.size
    cvis = np.zeros((nrows, nwave), dtype=np.complex128)
    for iw, lam in enumerate(wavelengths_m):
        u_lam = ucoord_m / lam
        v_lam = vcoord_m / lam
        cvis[:, iw] = np.asarray(cvis_fn(u_lam, v_lam, float(lam)), dtype=np.complex128)
    return cvis


def _add_noise_and_errors(
    values: np.ndarray,
    sigma: float,
    rng: Optional[np.random.Generator],
    add_noise: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create constant error array and optionally add Gaussian noise."""
    err = np.full_like(values, sigma, dtype=float)
    if add_noise and rng is not None:
        values = values + rng.normal(0.0, sigma, size=values.shape)
    return values, err


def sample_model_observables(
    model_cvis: ComplexVisibilityModel,
    sampling: Dict[str, np.ndarray],
    wavelengths_m: np.ndarray,
    noise: NoiseConfig = NoiseConfig(),
) -> Dict[str, np.ndarray]:
    """
    Sample a complex visibility model at UV points and produce OIFITS observables.

    Args:
        model_cvis: Callable returning complex visibility.
            Signature: model_cvis(u_lam, v_lam, wavelength_m) -> complex array.
        sampling: Output of generate_uv_sampling().
        wavelengths_m: (nwave,) effective wavelengths in meters.
        noise: Noise/error configuration.

    Returns:
        Dict containing VIS/VIS2/T3 data arrays ready for writing.
    """
    wavelengths_m = np.asarray(wavelengths_m, dtype=float)
    rng = np.random.default_rng(noise.seed) if noise.add_noise else None

    cvis_fn = model_cvis

    vis_cvis = _evaluate_cvis_grid(
        cvis_fn,
        sampling["vis_ucoord"],
        sampling["vis_vcoord"],
        wavelengths_m,
    )
    visamp = np.abs(vis_cvis)
    visphi = np.angle(vis_cvis, deg=True)
    vis2 = visamp**2

    visamp, visamp_err = _add_noise_and_errors(visamp, noise.visamp_err, rng, noise.add_noise)
    visphi, visphi_err = _add_noise_and_errors(visphi, noise.visphi_err_deg, rng, noise.add_noise)
    vis2, vis2_err = _add_noise_and_errors(vis2, noise.v2_err, rng, noise.add_noise)

    cvis1 = _evaluate_cvis_grid(
        cvis_fn,
        sampling["t3_u1coord"],
        sampling["t3_v1coord"],
        wavelengths_m,
    )
    cvis2 = _evaluate_cvis_grid(
        cvis_fn,
        sampling["t3_u2coord"],
        sampling["t3_v2coord"],
        wavelengths_m,
    )
    cvis3 = _evaluate_cvis_grid(
        cvis_fn,
        sampling["t3_u1coord"] + sampling["t3_u2coord"],
        sampling["t3_v1coord"] + sampling["t3_v2coord"],
        wavelengths_m,
    )
    t3 = cvis1 * cvis2 * cvis3
    t3amp = np.abs(t3)
    t3phi = np.angle(t3, deg=True)
    t3amp, t3amp_err = _add_noise_and_errors(t3amp, noise.t3amp_err, rng, noise.add_noise)
    t3phi, t3phi_err = _add_noise_and_errors(t3phi, noise.t3phi_err_deg, rng, noise.add_noise)

    return {
        "visamp": visamp,
        "visamp_err": visamp_err,
        "visphi": visphi,
        "visphi_err": visphi_err,
        "vis2data": vis2,
        "vis2err": vis2_err,
        "t3amp": t3amp,
        "t3amp_err": t3amp_err,
        "t3phi": t3phi,
        "t3phi_err": t3phi_err,
    }


def _mk_oi_target_hdu(target_name: str, target_id: int = 1) -> fits.BinTableHDU:
    cols = [
        fits.Column(name="TARGET_ID", format="I", array=np.array([target_id], dtype=np.int16)),
        fits.Column(name="TARGET", format="32A", array=np.array([target_name], dtype="S32")),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="OI_TARGET")
    hdu.header["OI_REVN"] = 2
    return hdu


def _mk_oi_wavelength_hdu(
    wavelengths_m: np.ndarray,
    eff_band_m: np.ndarray,
    insname: str,
) -> fits.BinTableHDU:
    cols = [
        fits.Column(name="EFF_WAVE", format="D", array=wavelengths_m.astype(float)),
        fits.Column(name="EFF_BAND", format="D", array=eff_band_m.astype(float)),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="OI_WAVELENGTH")
    hdu.header["OI_REVN"] = 2
    hdu.header["INSNAME"] = insname
    return hdu


def _mk_oi_vis_hdu(
    sampling: Dict[str, np.ndarray],
    obs: Dict[str, np.ndarray],
    target_id: int,
    insname: str,
) -> fits.BinTableHDU:
    nrows, nwave = obs["visamp"].shape
    cols = [
        fits.Column(name="TARGET_ID", format="I", array=np.full(nrows, target_id, dtype=np.int16)),
        fits.Column(name="MJD", format="D", array=sampling["vis_mjd"]),
        fits.Column(name="UCOORD", format="D", array=sampling["vis_ucoord"]),
        fits.Column(name="VCOORD", format="D", array=sampling["vis_vcoord"]),
        fits.Column(name="VISAMP", format=f"{nwave}D", array=obs["visamp"]),
        fits.Column(name="VISAMPERR", format=f"{nwave}D", array=obs["visamp_err"]),
        fits.Column(name="VISPHI", format=f"{nwave}D", array=obs["visphi"]),
        fits.Column(name="VISPHIERR", format=f"{nwave}D", array=obs["visphi_err"]),
        fits.Column(name="STA_INDEX", format="2I", array=sampling["vis_sta_index"]),
        fits.Column(name="FLAG", format=f"{nwave}L", array=np.zeros((nrows, nwave), dtype=bool)),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="OI_VIS")
    hdu.header["OI_REVN"] = 2
    hdu.header["INSNAME"] = insname
    return hdu


def _mk_oi_vis2_hdu(
    sampling: Dict[str, np.ndarray],
    obs: Dict[str, np.ndarray],
    target_id: int,
    insname: str,
) -> fits.BinTableHDU:
    nrows, nwave = obs["vis2data"].shape
    cols = [
        fits.Column(name="TARGET_ID", format="I", array=np.full(nrows, target_id, dtype=np.int16)),
        fits.Column(name="MJD", format="D", array=sampling["vis_mjd"]),
        fits.Column(name="UCOORD", format="D", array=sampling["vis_ucoord"]),
        fits.Column(name="VCOORD", format="D", array=sampling["vis_vcoord"]),
        fits.Column(name="VIS2DATA", format=f"{nwave}D", array=obs["vis2data"]),
        fits.Column(name="VIS2ERR", format=f"{nwave}D", array=obs["vis2err"]),
        fits.Column(name="STA_INDEX", format="2I", array=sampling["vis_sta_index"]),
        fits.Column(name="FLAG", format=f"{nwave}L", array=np.zeros((nrows, nwave), dtype=bool)),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="OI_VIS2")
    hdu.header["OI_REVN"] = 2
    hdu.header["INSNAME"] = insname
    return hdu


def _mk_oi_t3_hdu(
    sampling: Dict[str, np.ndarray],
    obs: Dict[str, np.ndarray],
    target_id: int,
    insname: str,
) -> fits.BinTableHDU:
    nrows, nwave = obs["t3amp"].shape
    cols = [
        fits.Column(name="TARGET_ID", format="I", array=np.full(nrows, target_id, dtype=np.int16)),
        fits.Column(name="MJD", format="D", array=sampling["t3_mjd"]),
        fits.Column(name="U1COORD", format="D", array=sampling["t3_u1coord"]),
        fits.Column(name="V1COORD", format="D", array=sampling["t3_v1coord"]),
        fits.Column(name="U2COORD", format="D", array=sampling["t3_u2coord"]),
        fits.Column(name="V2COORD", format="D", array=sampling["t3_v2coord"]),
        fits.Column(name="T3AMP", format=f"{nwave}D", array=obs["t3amp"]),
        fits.Column(name="T3AMPERR", format=f"{nwave}D", array=obs["t3amp_err"]),
        fits.Column(name="T3PHI", format=f"{nwave}D", array=obs["t3phi"]),
        fits.Column(name="T3PHIERR", format=f"{nwave}D", array=obs["t3phi_err"]),
        fits.Column(name="STA_INDEX", format="3I", array=sampling["t3_sta_index"]),
        fits.Column(name="FLAG", format=f"{nwave}L", array=np.zeros((nrows, nwave), dtype=bool)),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="OI_T3")
    hdu.header["OI_REVN"] = 2
    hdu.header["INSNAME"] = insname
    return hdu


def create_oifits_from_model(
    output_path: str,
    model_cvis: ComplexVisibilityModel,
    station_enu_m: np.ndarray,
    hour_angles_rad: Iterable[float],
    dec_rad: float,
    wavelengths_m: Iterable[float],
    eff_band_m: Optional[Iterable[float]] = None,
    target_name: str = "SYNTH_TARGET",
    target_id: int = 1,
    insname: str = "SYNTH_INS",
    mjd_start: float = 60000.0,
    cadence_sec: float = 600.0,
    noise: NoiseConfig = NoiseConfig(),
    overwrite: bool = True,
) -> str:
    """
    Create an OIFITS file by sampling a visibility model on synthetic UV coverage.

    Args:
        output_path: Output OIFITS path.
        model_cvis: Callable model_cvis(u_lam, v_lam, wavelength_m) -> complex array.
        station_enu_m: (nstations, 3) station coordinates in ENU meters.
        hour_angles_rad: Iterable of hour angles in radians.
        dec_rad: Target declination in radians.
        wavelengths_m: Effective wavelengths in meters.
        eff_band_m: Bandwidth per channel (meters). Defaults to 1% of wavelength.
        target_name: OI_TARGET name.
        target_id: OI_TARGET id.
        insname: Instrument name for OI_WAVELENGTH and observables.
        mjd_start: MJD at first sample.
        cadence_sec: Time spacing between samples.
        noise: NoiseConfig for errors and optional noise.
        overwrite: Overwrite output if it exists.

    Returns:
        output_path
    """
    hour_angles_rad = np.asarray(list(hour_angles_rad), dtype=float)
    wavelengths_m = np.asarray(list(wavelengths_m), dtype=float)
    if hour_angles_rad.size == 0:
        raise ValueError("hour_angles_rad cannot be empty")
    if wavelengths_m.size == 0:
        raise ValueError("wavelengths_m cannot be empty")

    if eff_band_m is None:
        eff_band_m = 0.01 * wavelengths_m
    eff_band_m = np.asarray(list(eff_band_m), dtype=float)
    if eff_band_m.shape != wavelengths_m.shape:
        raise ValueError("eff_band_m must have same shape as wavelengths_m")

    mjd_times = mjd_start + np.arange(hour_angles_rad.size) * cadence_sec / 86400.0
    sampling = generate_uv_sampling(
        station_enu_m=station_enu_m,
        hour_angles_rad=hour_angles_rad,
        dec_rad=dec_rad,
        mjd_times=mjd_times,
    )
    obs = sample_model_observables(
        model_cvis=model_cvis,
        sampling=sampling,
        wavelengths_m=wavelengths_m,
        noise=noise,
    )

    hdus = fits.HDUList(
        [
            fits.PrimaryHDU(),
            _mk_oi_target_hdu(target_name=target_name, target_id=target_id),
            _mk_oi_wavelength_hdu(wavelengths_m=wavelengths_m, eff_band_m=eff_band_m, insname=insname),
            _mk_oi_vis_hdu(sampling=sampling, obs=obs, target_id=target_id, insname=insname),
            _mk_oi_vis2_hdu(sampling=sampling, obs=obs, target_id=target_id, insname=insname),
            _mk_oi_t3_hdu(sampling=sampling, obs=obs, target_id=target_id, insname=insname),
        ]
    )
    hdus.writeto(output_path, overwrite=overwrite)
    return output_path


__all__ = [
    "NoiseConfig",
    "project_baseline_to_uv",
    "generate_uv_sampling",
    "sample_model_observables",
    "create_oifits_from_model",
]
