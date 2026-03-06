import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def generate_ring_image(npix=128, diameter_px=64.0, width_px=8.0, center=None, outside=0.0, inside=1.0):
    """Generate a simple 2D smooth ring image with a Gaussian radial profile."""
    if diameter_px <= 0:
        raise ValueError("diameter_px must be > 0")
    if width_px <= 0:
        raise ValueError("width_px must be > 0")

    if center is None:
        cx = cy = (npix - 1) / 2.0
    else:
        cx, cy = center

    y, x = np.indices((npix, npix), dtype=float)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring_radius = diameter_px / 2.0
    sigma = width_px / 2.3548200450309493  # convert FWHM-like width to sigma
    profile = np.exp(-0.5 * ((r - ring_radius) / sigma) ** 2)
    image = float(outside) + (float(inside) - float(outside)) * profile
    return image


def main():
    parser = argparse.ArgumentParser(description="Generate a simple uniform ring image.")
    parser.add_argument("--npix", type=int, default=128, help="Image size (npix x npix).")
    parser.add_argument("--diameter", type=float, default=64.0, help="Ring diameter in pixels (profile peak).")
    parser.add_argument("--width", type=float, default=8.0, help="Ring radial FWHM in pixels.")
    parser.add_argument("--output", type=str, default="ring.npy", help="Output .npy filename.")
    args = parser.parse_args()

    img = generate_ring_image(npix=args.npix, diameter_px=args.diameter, width_px=args.width)
    np.save(args.output, img)
    png_path = str(Path(args.output).with_suffix(".png"))
    plt.imsave(png_path, img, cmap="gray")
    print(f"Saved ring image to {args.output} and {png_path} with shape {img.shape}")


if __name__ == "__main__":
    main()
