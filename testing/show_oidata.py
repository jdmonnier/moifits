#!/usr/bin/env python3
"""
Small helper to load an OIFITS file and print the OIData summary.

Usage:
    python moifits/testing/show_oidata.py /path/to/file.oifits
"""

import argparse
import sys
from pathlib import Path

from astropy.io import fits

# Import local readoifits.py directly (avoids full package import requirements).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from readoifits import readoifits  # noqa: E402


def print_hdu_table_list(oifits_file: Path) -> None:
    """Print available HDUs/tables in the OIFITS file."""
    with fits.open(oifits_file) as hdul:
        print(f"File: {oifits_file}")
        print(f"Number of HDUs: {len(hdul)}")
        print("")
        for i, hdu in enumerate(hdul):
            rows = "-"
            if hdu.data is not None:
                try:
                    rows = len(hdu.data)
                except Exception:
                    rows = "-"
            print(f"{i:2d}: {hdu.name:<15} rows={rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load an OIFITS file and print OIData.")
    parser.add_argument("oifits_file", type=Path, help="Path to OIFITS file")
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

    print_hdu_table_list(args.oifits_file)
    print("")

    data = readoifits(
        str(args.oifits_file),
        filter_bad_data=not args.no_filter,
        redundance_remove=not args.no_redundance_remove,
    )
    print(data)


if __name__ == "__main__":
    main()
