#!/usr/bin/env python3
"""
Plot OIData products from an OIFITS file.

Usage:
    moifits-plot /path/to/file.oifits
    moifits-plot /path/to/file.oifits --save overview.png
"""

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot VIS/VIS2/T3 and UV coverage from OIFITS.")
    parser.add_argument("oifits_file", type=Path, help="Path to OIFITS file")
    parser.add_argument(
        "--color-by",
        choices=["wavelength", "mjd", "none"],
        default="wavelength",
        help="Color points by wavelength or MJD",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        default=True,
        help="Show error bars in baseline plots (default: on)",
    )
    parser.add_argument(
        "--no-errors",
        action="store_false",
        dest="show_errors",
        help="Disable error bars in baseline plots",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Output image path. If omitted, shows interactive window.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable bad-data filtering while reading",
    )
    parser.add_argument(
        "--no-redundance-remove",
        action="store_true",
        help="Disable redundant UV point removal while reading",
    )
    args = parser.parse_args()

    if not args.oifits_file.exists():
        raise FileNotFoundError(f"OIFITS file not found: {args.oifits_file}")

    import matplotlib.pyplot as plt

    from moifits.plot_oifits import plot_observables_overview
    from moifits.readoifits import readoifits

    color_by = None if args.color_by == "none" else args.color_by
    t0 = time.perf_counter()
    data = readoifits(
        str(args.oifits_file),
        filter_bad_data=not args.no_filter,
        redundance_remove=not args.no_redundance_remove,
    )
    print(f"[timing] readoifits: {time.perf_counter() - t0:.3f} s")
    print(
        "[counts] "
        f"VISAMP: {data.nvisamp} | VISPHI: {data.nvisphi} | "
        f"VIS2: {data.nv2} | T3AMP: {data.nt3amp} | T3PHI: {data.nt3phi}"
    )
    fig, _ = plot_observables_overview(
        data,
        color_by=color_by,
        show_errors=args.show_errors,
        show_conjugate_uv=True,
    )
    fig.suptitle(str(args.oifits_file), fontsize=10)
    fig.tight_layout()

    if args.save is not None:
        fig.savefig(args.save, dpi=180, bbox_inches="tight")
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
