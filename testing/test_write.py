import numpy as np
import sys
from pathlib import Path
import argparse
import time

# Import local modules directly from moifits/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from writeoifits import generate_uv_sampling
from image_to_observables import image_to_cvis_grid, sample_image_observables, create_oifits_from_image

def main():
    parser = argparse.ArgumentParser(description="Create synthetic OIFITS from a .npy image.")
    parser.add_argument("--image", type=str, default="ring.npy", help="Input .npy image path.")
    parser.add_argument("--output", type=str, default="synthetic_from_image.oifits", help="Output OIFITS path.")
    parser.add_argument("--pixsize", type=float, default=0.125, help="Pixel scale in mas.")
    args = parser.parse_args()

    t0_total = time.perf_counter()

    # Configuration: add two longer baselines, sample every 30 s for 3 h, 50 channels.
    stations = np.array(
        [
            [0, 0, 0],
            [40, 0, 0],
            [0, 60, 0],
            [55, 55, 0],
            [180, 0, 0],   # longer baseline station
            [0, 220, 0],   # longer baseline station
        ],
        dtype=float,
    )
    cadence_sec = 30.0
    duration_sec = 3.0 * 3600.0
    time_sec = np.arange(0.0, duration_sec + cadence_sec, cadence_sec)
    sidereal_day_sec = 86164.0905
    ha = (time_sec - 0.5 * duration_sec) * (2.0 * np.pi / sidereal_day_sec)
    mjd_start = 60000.0
    mjd = mjd_start + time_sec / 86400.0

    t0 = time.perf_counter()
    sampling = generate_uv_sampling(stations, ha, dec_rad=np.deg2rad(30.0), mjd_times=mjd)
    t_sampling = time.perf_counter() - t0
    print(f"[timing] UV sampling: {t_sampling:.3f} s")

    # 2) Load image -> cvis
    t0 = time.perf_counter()
    image = np.load(args.image).astype(float)
    waves = np.linspace(1.60e-6, 1.70e-6, 50)
    _ = image_to_cvis_grid(
        image,
        pixsize_mas=args.pixsize,
        ucoord_m=sampling["vis_ucoord"],
        vcoord_m=sampling["vis_vcoord"],
        wavelengths_m=waves,
    )
    t_cvis = time.perf_counter() - t0
    print(f"[timing] image_to_cvis_grid: {t_cvis:.3f} s")

    # 3) Load image -> observables
    t0 = time.perf_counter()
    obs = sample_image_observables(image, pixsize_mas=args.pixsize, sampling=sampling, wavelengths_m=waves)
    t_obs = time.perf_counter() - t0
    print(f"[timing] sample_image_observables: {t_obs:.3f} s")

    # 4) Load image -> OIFITS
    t0 = time.perf_counter()
    create_oifits_from_image(
        args.output,
        image=image,
        pixsize_mas=args.pixsize,
        station_enu_m=stations,
        hour_angles_rad=ha,
        dec_rad=np.deg2rad(30.0),
        wavelengths_m=waves,
        mjd_start=mjd_start,
        cadence_sec=cadence_sec,
    )
    t_write = time.perf_counter() - t0
    print(f"[timing] create_oifits_from_image: {t_write:.3f} s")

    total_points = (
        obs["visamp"].size
        + obs["visphi"].size
        + obs["vis2data"].size
        + obs["t3amp"].size
        + obs["t3phi"].size
    )
    t_total = time.perf_counter() - t0_total
    print(f"[timing] total: {t_total:.3f} s")
    print(f"Wrote {args.output} from {args.image} (shape={image.shape}, pixsize={args.pixsize} mas)")
    print(f"Total points: {total_points}")


if __name__ == "__main__":
    main()
