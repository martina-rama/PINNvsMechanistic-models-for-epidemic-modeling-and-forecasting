"""
load_matrices.py
----------------
Load the contact matrix (Prem et al.) of a given country from the files
MUestimates_all_locations_1.xlsx and/or MUestimates_all_locations_2.xlsx.

Structure of the files:
  - Each sheet is a country.
  - File _1: 17 x 16 sheet, the first row is a header (X1, ..., X16) that is discarded.
  - File _2: 16 x 16 sheet, the matrix only.

Quick use
---------
    from pinnvsmech.data.load_matrices import load_contact_matrix
    C = load_contact_matrix("Italy")
    print(C.shape)   # (16, 16)

    # Fuzzy search: similar names are suggested
    C = load_contact_matrix("united kingdom")
    # or list all the available countries
    from pinnvsmech.data.load_matrices import list_countries
    print(list_countries())
"""

import os
import numpy as np
import pandas as pd
from difflib import get_close_matches
from typing import Union

from pinnvsmech.paths import EPI_DIR


_DEFAULT_DATA_DIR = str(EPI_DIR / "contact_matrices_152_countries")
_FILE_PATTERN = "MUestimates_all_locations_{n}.xlsx"
_N_FILES = 2          # files _1 and _2; increase if there are more files


def _xlsx_files(data_dir: str) -> list[str]:
    """Return the list of MUestimates files found in the folder."""
    files = []
    for n in range(1, _N_FILES + 1):
        path = os.path.join(data_dir, _FILE_PATTERN.format(n=n))
        if os.path.isfile(path):
            files.append(path)
    if not files:
        raise FileNotFoundError(
            f"No MUestimates_all_locations_*.xlsx file found in '{data_dir}'.\n"
            f"Set the data_dir parameter to the correct path."
        )
    return files


def list_countries(data_dir: str = _DEFAULT_DATA_DIR) -> list[str]:
    """Return the list of all the countries available in the xlsx files."""
    countries = []
    for path in _xlsx_files(data_dir):
        xl = pd.ExcelFile(path)
        countries.extend(xl.sheet_names)
    return sorted(countries)


def _find_country(name: str, available: list[str]) -> str:
    """
    Look for the country name (case-insensitive, fuzzy).
    Return the exact name of the sheet, or raise ValueError with suggestions.
    """
    name_lower = name.strip().lower()
    for c in available:
        if c.lower() == name_lower:
            return c

    matches = get_close_matches(name_lower, [c.lower() for c in available],
                                n=5, cutoff=0.5)
    if matches:
        suggestions = [c for c in available if c.lower() in matches]
        raise ValueError(
            f"Country '{name}' not found.\n"
            f"Did you mean one of these?\n  " + "\n  ".join(suggestions)
        )
    raise ValueError(
        f"Country '{name}' not found and no similar name.\n"
        f"Use list_countries() to see all the available countries."
    )


def load_contact_matrix(
    country: str,
    data_dir: str = _DEFAULT_DATA_DIR,
    as_dataframe: bool = False,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Load the 16x16 contact matrix of the given country.

    Parameters
    ----------
    country : str
        Name of the country (e.g. "Italy", "United Kingdom of Great Britain").
        The search is case-insensitive and tolerates small variations.
    data_dir : str
        Folder containing the files MUestimates_all_locations_*.xlsx.
        Default: <repo>/data/epi/contact_matrices_152_countries/
    as_dataframe : bool
        If True returns a pd.DataFrame with index/columns X1..X16;
        if False (default) returns an np.ndarray (16, 16).

    Returns
    -------
    np.ndarray or pd.DataFrame  (16, 16)

    Raises
    ------
    FileNotFoundError : xlsx file not found in data_dir
    ValueError        : country not found (with fuzzy suggestions)
    """
    files = _xlsx_files(data_dir)
    available = []
    for path in files:
        xl = pd.ExcelFile(path)
        available.extend(xl.sheet_names)

    exact_name = _find_country(country, available)

    for path in files:
        xl = pd.ExcelFile(path)
        if exact_name in xl.sheet_names:
            raw = pd.read_excel(path, sheet_name=exact_name, header=None)

            if raw.shape == (17, 16):
                df = raw.iloc[1:, :].copy()

            elif raw.shape == (16, 16):
                df = raw.copy()

            else:
                raise ValueError(
                    f"Unexpected shape for '{exact_name}': {raw.shape}"
                )

            df.columns = [f"X{i}" for i in range(1, df.shape[1] + 1)]
            df.index   = [f"X{i}" for i in range(1, df.shape[0] + 1)]
            break

    if df.shape != (16, 16):
        raise ValueError(
            f"Unexpected shape for '{exact_name}': {df.shape} (expected 16x16)."
        )

    df.index.name = None
    df.columns.name = None

    if as_dataframe:
        return df
    return df.to_numpy(dtype=float)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the Prem et al. contact matrix of a country."
    )
    parser.add_argument("country", nargs="?", help="Name of the country")
    parser.add_argument("--data-dir", default=_DEFAULT_DATA_DIR,
                        help="Folder with the xlsx files")
    parser.add_argument("--list", action="store_true",
                        help="List all the available countries")
    args = parser.parse_args()

    if args.list:
        countries = list_countries(args.data_dir)
        print(f"{len(countries)} countries available:")
        for c in countries:
            print(f"  {c}")
        sys.exit(0)

    if not args.country:
        parser.print_help()
        sys.exit(1)

    try:
        C = load_contact_matrix(args.country, data_dir=args.data_dir)
        df_out = pd.DataFrame(
            C,
            index=[f"X{i}" for i in range(1, 17)],
            columns=[f"X{i}" for i in range(1, 17)],
        )
        print(f"\nContact matrix for '{args.country}' ({C.shape[0]}x{C.shape[1]}):\n")
        print(df_out.to_string(float_format="{:.6f}".format))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
