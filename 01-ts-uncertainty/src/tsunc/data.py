"""Extract hourly Météo-France series: one station, a few variables, a time range.

Source: data.gouv.fr, dataset "Données climatologiques de base - horaires" (Licence Ouverte 2.0).
The dataset has one compressed CSV (an "archive") per département and period (about a decade).
Each archive holds ALL the stations of the département and about 190 columns; times are UTC.

How the module is organised (read it top to bottom):

    1. Which archives exist?     list_files, files_from_urls, select_files
    2. Download                  download_file, download_files
    3. Read an archive           list_stations, read_station
    4. Small cached extracts     extract_one_file
    5. Build the hourly table    build_frame (from archives), extract (from extracts)
    6. Check and cut             gap_report, valid_per_year, make_blocks

Everything is one call away with ``extract``::

    frame = extract("31069001", variables=["T"], start="2010", end="2023")

What ``extract`` does for each archive: download it, keep only the rows of the station and the
columns of the variables (the "extract", a file of about a megabyte), save the extract, delete the
archive. The next call finds the extracts and downloads nothing.

Terms used below:
    archive  : a big file downloaded from the internet (one département, ten years).
    extract  : the small file we keep from an archive (one station, a few variables).
    NaN      : "not a number", pandas' marker for a missing value (``NA`` in R).
    UTC      : universal time; no daylight-saving jumps.
    
    
Typical use (from the project folder)::

    frame = extract("31069001", variables=["T", "U"], start="1990", end="2025")

or, step by step::

    files = list_files("31")                       # table of the archives of a département
    files = select_files(files, "1990", "2025")    # keep the archives that overlap the range
    paths = download_files(files, "data/raw")      # saved once, reused afterwards
    frame = build_frame(paths, "31069001", ["T", "U"], "1990", "2025")
    blocks = make_blocks(frame)                    # train / val / calibration / test

All functions take explicit paths: nothing assumes where the repository lives.
"""


# ============================================================
# PYTHON STANDARD LIBRARIES
# These libraries are included with Python and do not need
# to be installed separately.
# ============================================================

import gzip          # Opens compressed .gz files from Météo-France.
import itertools     # Provides tools for working with sequences.
# Here, pairwise() compares consecutive time boundaries
# for the train, validation, calibration and test blocks.

import json          # Reads JSON data returned by the data.gouv.fr API.
# Here, json.load(response) interprets the API response.

import re             # Searches and replaces text using regular expressions.
# Here, it extracts years from archive names and
# makes downloaded file names safe to use.

import shutil         # Provides utilities for working with files.
# Here, it copies downloaded data to disk progressively.

import urllib.parse   # Parses Internet addresses (URLs).
# Here, it extracts file names from URLs and checks
# that download links use HTTPS.

import urllib.request # Sends HTTP requests.
# Here, it contacts data.gouv.fr and downloads archives.

import warnings       # Displays warnings without necessarily stopping the program.
# Here, it reports archives that contain no data
# for the requested weather station.

from collections.abc import Iterator, Sequence
# Iterator: describes an object that provides elements one at a time.
#           Here, _read_chunks() yields successive pandas DataFrames.
# Sequence: describes an ordered collection of elements.
#           Here, it represents a list of variables such as ["T", "U"].

from pathlib import Path
# Path: provides convenient tools for handling file and folder paths.
#       Here, it is used to create folders, check whether files exist,
#       and retrieve file names and extensions.


# ============================================================
# EXTERNAL LIBRARIES
# These libraries must be installed in the Python environment.
# ============================================================

import numpy as np
# NumPy: library for numerical computing.
#        Here, np.ndarray is used as a type annotation for an array
#        of Boolean values in the data quality checks.

import pandas as pd
# Pandas: library for working with structured data tables.
#         Here, it reads CSV files, cleans data, converts dates,
#         filters weather stations and combines hourly observations.


# --- Project settings (edit here) -------------------------------------------------------

# API endpoint used to retrieve the Météo-France dataset information.
DATASET_API = (
  "https://www.data.gouv.fr/api/1/datasets/donnees-climatologiques-de-base-horaires/"
)
DEPARTMENT = "31"  # Haute-Garonne
# NUM_POSTE of TOULOUSE-BLAGNAC (seen with list_stations on 2026-11-12). Not 31070004.
STATION_ID = "31069001"

# ============================================================
# TIME BLOCKS
# Define the chronological periods used for model development
# and evaluation. All dates are expressed in UTC.
# ============================================================

# Starting date of each block.
# The blocks are consecutive and do not overlap:
# - train: model training
# - val: model validation and hyperparameter selection
# - calibration: conformal prediction interval calibration
# - test: final model evaluation
BLOCK_STARTS = {
    "train": "2000-01-01",
    "val": "2021-01-01",
    "calibration": "2022-01-01",
    "test": "2023-01-01",
}
BLOCKS_END = "2024-01-01"


# Expected physical ranges for selected weather variables:
# T  : air temperature, in degrees Celsius.
# TD : dew point temperature, in degrees Celsius.
# U  : relative humidity, in percent.
#
# Plausible range of some variables. It catches a wrong unit or decimal mark (for instance
# tenths of degrees) instead of silently using it. Variables not listed are not checked.
PLAUSIBLE_RANGES = {"T": (-60.0, 60.0), "TD": (-60.0, 60.0), "U": (0.0, 100.0)}

# ============================================================
# SOURCE FILE COLUMN NAMES AND READING SETTINGS
# The original CSV files contain about 190 columns.
# Only the columns needed by the project are selected.
# ============================================================

# Name of the source column containing the observation date and hour.
# Example: "2004120106" means 2004-12-01 at 06:00 UTC.
_TIME = "AAAAMMJJHH"  # e.g. 2004120106 = 2004-12-01 06:00 UTC
_STATION_COLS = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI"]
_CHUNKSIZE = 200_000  # rows read at a time, so a big file never has to fit in memory
_USER_AGENT = {"User-Agent": "tsunc (research portfolio)"}


# =========================================================================================
# 1. Which archives exist, and which ones do we need?
# =========================================================================================


import gzip
import itertools
import json
import re
import shutil
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# --- Project settings (edit here) -------------------------------------------------------

DATASET_API = (
    "https://www.data.gouv.fr/api/1/datasets/donnees-climatologiques-de-base-horaires/"
)
DEPARTMENT = "31"  # Haute-Garonne
STATION_ID = (
    "31069001"  # NUM_POSTE of TOULOUSE-BLAGNAC (seen with list_stations); not 31070004
)

# First hour (UTC) of each block, then the end of the last block (excluded).
# Blocks are disjoint and in this order: train, val (early stopping, hyperparameters),
# calibration (conformal scores), test (final evaluation).
BLOCK_STARTS = {
    "train": "2010-01-01",
    "val": "2021-01-01",
    "calibration": "2022-01-01",
    "test": "2023-01-01",
}
BLOCKS_END = "2024-01-01"

# Plausible range of some variables. It catches a wrong unit or decimal mark (for instance
# tenths of degrees) instead of silently using it. Variables not listed are not checked.
PLAUSIBLE_RANGES = {"T": (-60.0, 60.0), "TD": (-60.0, 60.0), "U": (0.0, 100.0)}

# --- Names used in the source files (about 190 columns; we read only the ones we need) ---

_TIME = "AAAAMMJJHH"  # e.g. 2004120106 = 2004-12-01 06:00 UTC
_STATION_COLS = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI"]
_CHUNKSIZE = 200_000  # rows read at a time, so a big file never has to fit in memory
_USER_AGENT = {"User-Agent": "tsunc (research portfolio)"}


# =========================================================================================
# 1. Which archives exist, and which ones do we need?
# =========================================================================================


def _years_in_name(name: str) -> tuple[int, int] | None:
    """Find a period such as '2010-2019' in a file name.

    Args:
        name: a file name, e.g. "H_31_previous-2020-2024.csv.gz".

    Returns:
        (first_year, last_year), e.g. (2020, 2024), or None if the name holds no period.

    Example:
        >>> _years_in_name("H_31_2010-2019.csv.gz")
        (2010, 2019)
        >>> _years_in_name("README.pdf") is None
        True
    """
    match = re.search(r"(\d{4})-(\d{4})", name)  # four digits, a dash, four digits
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def files_from_urls(urls: dict[str, str]) -> pd.DataFrame:
    """Build the table of archives from a hand-made {file name: link} dictionary.

    Use it when list_files() cannot reach data.gouv.fr.

    Args:
        urls: file name -> https link. Each name must contain its period,
            e.g. {"H_31_2010-2019.csv.gz": "https://..."}.

    Returns:
        A table with one row per archive and the columns ``name``, ``url``, ``first_year``,
        ``last_year``, sorted from the oldest period to the newest.

    Raises:
        ValueError: a name holds no period like '2010-2019'.
    """
    rows = []
    for name, url in urls.items():
        years = _years_in_name(name)
        if years is None:
            raise ValueError(f"No period like '2010-2019' found in the name {name!r}")
        rows.append(
            {"name": name, "url": url, "first_year": years[0], "last_year": years[1]}
        )
    table = pd.DataFrame(rows, columns=["name", "url", "first_year", "last_year"])
    return table.sort_values(["first_year", "last_year"]).reset_index(drop=True)


def list_files(
    department: str = DEPARTMENT, api_url: str = DATASET_API
) -> pd.DataFrame:
    """Ask data.gouv.fr for the hourly archives of one département (one row per archive).

    Needs an internet connection. Checked on Colab on 2026-11-12: 24 archives for the
    département 31, from 1800-1809 to 2025-2026.

    Args:
        department: département number as text, e.g. "31".
        api_url: address of the dataset description (JSON) on data.gouv.fr.

    Returns:
        The same table as files_from_urls(): name, url, first_year, last_year.

    Raises:
        TypeError: the answer of the site has no list of "resources" (format changed).
        ValueError: no ``.csv.gz`` archive found for this département.
    """
    _require_https(api_url)
    request = urllib.request.Request(api_url, headers=_USER_AGENT)
    with urllib.request.urlopen(request, timeout=60) as response:
        dataset = json.load(response)  # the answer is JSON text -> Python dict
    resources = dataset.get("resources")
    if not isinstance(resources, list):
        raise TypeError(
            "Unexpected answer from data.gouv.fr (no list of 'resources'). "
            "Use files_from_urls() with links copied from the dataset page."
        )

    # "_31_" or "-31-" in the name; the pattern does not match "_310_" or "_131_".
    wanted = re.compile(rf"(^|[_-]){re.escape(department)}[_-]")
    urls = {}
    seen = []
    for resource in resources:
        url = str(resource.get("url") or "")
        url_name = Path(urllib.parse.urlparse(url).path).name
        # The file name may be the resource title or the last part of its link.
        for name in (str(resource.get("title") or ""), url_name):
            seen.append(name)
            if (
                name.endswith(".csv.gz")
                and wanted.search(name)
                and _years_in_name(name)
            ):
                urls[name] = url
                break
    if not urls:
        raise ValueError(
            f"No .csv.gz file found for département {department!r}. "
            f"Some names seen: {seen[:10]}. Use files_from_urls() instead."
        )
    return files_from_urls(urls)


def select_files(files: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Keep the archives whose period overlaps [start, end].

    Args:
        files: table from list_files() or files_from_urls().
        start: first year or date wanted, e.g. "2010" or "2010-06-01"; None = from the oldest.
        end: last year or date wanted; None = up to the newest.

    Returns:
        The rows of ``files`` that overlap the range, in the same order.

    Raises:
        ValueError: no archive covers the range.

    Example: with archives 2000-2009, 2010-2019 and 2020-2024, select_files(files, "2015",
    "2021") keeps the last two.
    """
    first = files["first_year"].min() if start is None else pd.Timestamp(start).year
    last = files["last_year"].max() if end is None else pd.Timestamp(end).year
    # Two periods overlap when each one starts before the other ends.
    chosen = files[(files["first_year"] <= last) & (files["last_year"] >= first)]
    if chosen.empty:
        raise ValueError(
            f"No file covers {first}-{last}; the files cover "
            f"{files['first_year'].min()}-{files['last_year'].max()}."
        )
    return chosen


# =========================================================================================
# 2. Download
# =========================================================================================


def _require_https(url: str) -> None:
    """Stop with an error unless ``url`` starts with https:// (refuses file://, http://...)."""
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError(f"Only https links are accepted, got {url!r}")


def download_file(url: str, dest: str | Path, overwrite: bool = False) -> Path:
    """Download ``url`` to ``dest`` unless the file is already there.

    The download goes to a temporary ``.part`` file; it is renamed to ``dest`` only after a
    check that it is a gzip file. An interrupted download or an HTML error page is therefore
    never kept as if it were data.

    Args:
        url: https link of a ``.csv.gz`` file.
        dest: where to save it; missing folders are created.
        overwrite: True to download again even if ``dest`` exists.

    Returns:
        ``dest`` as a Path.

    Raises:
        ValueError: the link is not https.
        RuntimeError: the answer is not a gzip file.
    """
    _require_https(url)
    dest = Path(dest)
    if dest.exists() and not overwrite:
        return dest  # cache: nothing to do
    dest.parent.mkdir(parents=True, exist_ok=True)  # create the folder(s) if needed
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers=_USER_AGENT)
    try:
        with (
            urllib.request.urlopen(request, timeout=300) as response,
            open(tmp, "wb") as out,
        ):
            shutil.copyfileobj(response, out)  # copy the stream to disk, piece by piece
        with open(tmp, "rb") as handle:
            if handle.read(2) != b"\x1f\x8b":  # first two bytes of every gzip file
                raise RuntimeError(
                    f"{url} did not return a gzip file (an error page?). Open the link in a "
                    "browser to check that it still points to a data file."
                )
        tmp.replace(dest)  # rename: from now on the file is complete
    finally:
        tmp.unlink(missing_ok=True)  # remove the .part file if something went wrong
    return dest


def _safe_file_name(name: str) -> str:
    """Make a name coming from the internet safe to use as a file name.

    Any character other than letters, digits, ``.``, ``_`` and ``-`` becomes ``_``. A name that
    then starts with ``.`` (such as ``../x``) or does not end with ``.csv.gz`` is refused, so a
    booby-trapped link cannot write outside the chosen folder.

    Raises:
        ValueError: unexpected name.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if safe.startswith(".") or not safe.endswith(".csv.gz"):
        raise ValueError(f"Unexpected file name {name!r}")
    return safe


def download_files(files: pd.DataFrame, raw_dir: str | Path) -> list[Path]:
    """Download every archive of a table into ``raw_dir`` (already present files are kept).

    Args:
        files: table with the columns ``name`` and ``url`` (see select_files).
        raw_dir: folder for the archives, e.g. "data/raw" (ignored by Git).

    Returns:
        The paths of the archives, in the order of the table.
    """
    raw_dir = Path(raw_dir)
    return [
        download_file(row.url, raw_dir / _safe_file_name(row.name))
        for row in files.itertuples()
    ]


# =========================================================================================
# 3. Read an archive
# =========================================================================================


def _detect_separator(path: str | Path) -> str:
    """Guess the column separator of a CSV from its first line.

    Counts ';', tab and ',' in the header line; the most frequent one is the separator. This
    avoids guessing wrong when the real separator is not known in advance.

    Args:
        path: a ``.csv`` or ``.csv.gz`` file.

    Returns:
        One of ";", "\\t", ",".

    Raises:
        ValueError: none of the three appears in the first line.
    """
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    counts = {sep: header.count(sep) for sep in (";", "\t", ",")}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise ValueError(
            f"{Path(path).name}: cannot find a separator in {header[:80]!r}"
        )
    return best


def _read_chunks(
    path: str | Path,
    variables: Sequence[str],
    sep: str | None = None,
    station_id: str | None = None,
) -> Iterator[pd.DataFrame]:
    """Read an archive piece by piece and yield cleaned pieces (a "generator").

    Why pieces: an archive is hundreds of MB once decompressed; reading it whole could fill
    the memory of a Colab machine. Each piece has ``_CHUNKSIZE`` rows.

    Args:
        path: archive (``.csv.gz``).
        variables: column names to keep, e.g. ["T", "U"].
        sep: column separator; None = detected from the header.
        station_id: if given, rows of other stations are dropped BEFORE any conversion
            (much faster: most rows belong to other stations).

    Yields:
        A DataFrame per piece with the columns NUM_POSTE, NOM_USUEL, LAT, LON, ALTI (text or
        numbers), one float column per variable, and ``time`` (datetime in UTC).

    Raises:
        ValueError: a wanted column is not in the file (the message lists the first columns found).
    """
    sep = _detect_separator(path) if sep is None else sep
    wanted = [*_STATION_COLS, _TIME, *variables]
    # nrows=0 reads only the header line.
    header = pd.read_csv(path, sep=sep, nrows=0, encoding_errors="replace")
    missing = [c for c in wanted if c not in header.columns]
    if missing:
        raise ValueError(
            f"{Path(path).name}: missing columns {missing}. First columns found: "
            f"{list(header.columns)[:8]} (separator used: {sep!r})."
        )

    reader = pd.read_csv(
        path,
        sep=sep,
        usecols=wanted,  # read only these columns
        dtype=str,  # read everything as text first: no silent guess about types
        chunksize=_CHUNKSIZE,  # makes read_csv return an iterator of pieces
        encoding_errors="replace",
    )
    for chunk in reader:
        chunk["NUM_POSTE"] = chunk["NUM_POSTE"].fillna("").str.strip()
        if station_id is not None:
            chunk = chunk[chunk["NUM_POSTE"] == station_id]  # keep one station
        out = chunk[["NUM_POSTE", "NOM_USUEL"]].copy()
        out["NOM_USUEL"] = out["NOM_USUEL"].fillna("").str.strip()
        for column in ["LAT", "LON", "ALTI", *variables]:
            # decimal comma -> point, then text -> number; unreadable text becomes NaN
            text = chunk[column].fillna("").str.replace(",", ".", regex=False)
            out[column] = pd.to_numeric(text, errors="coerce")
        out["time"] = pd.to_datetime(
            chunk[_TIME], format="%Y%m%d%H", utc=True, errors="coerce"
        )  # "2004120106" -> 2004-12-01 06:00 UTC
        yield out.dropna(subset=["time"])


def list_stations(
    path: str | Path,
    variable: str = "T",
    name_contains: str | None = None,
    sep: str | None = None,
) -> pd.DataFrame:
    """One row per station of an archive: name, position, number of values of ``variable``.

    Use it on one archive to check a station number (the name shown is the reference).

    Args:
        path: archive (``.csv.gz``).
        variable: column whose non-empty values are counted, default "T".
        name_contains: keep the stations whose name matches this regular expression, not
            case-sensitive, e.g. "BLAGNAC|TOULOUSE".
        sep: column separator; None = detected.

    Returns:
        A table indexed by NUM_POSTE with NOM_USUEL, LAT, LON, ALTI, first_time, last_time
        (first and last hour with a value) and n_valid (number of values), sorted by n_valid
        in decreasing order.
    """
    pieces = []
    for chunk in _read_chunks(path, [variable], sep):
        valid = chunk.dropna(subset=[variable])
        grouped = valid.groupby("NUM_POSTE")["time"]
        by_station = chunk.groupby("NUM_POSTE")
        pieces.append(
            pd.DataFrame(
                {
                    "NOM_USUEL": by_station["NOM_USUEL"].first(),
                    "LAT": by_station["LAT"].first(),
                    "LON": by_station["LON"].first(),
                    "ALTI": by_station["ALTI"].first(),
                    "first_time": grouped.min(),
                    "last_time": grouped.max(),
                    "n_valid": grouped.size(),
                }
            )
        )
    if not pieces:
        raise ValueError(f"{Path(path).name}: no readable row")
    # Combine the pieces: a station appears in several pieces.
    table = (
        pd.concat(pieces)
        .groupby(level=0)
        .agg(
            {
                "NOM_USUEL": "first",
                "LAT": "first",
                "LON": "first",
                "ALTI": "first",
                "first_time": "min",
                "last_time": "max",
                "n_valid": "sum",
            }
        )
    )
    table["n_valid"] = table["n_valid"].fillna(0).astype(int)
    if name_contains is not None:
        keep = table["NOM_USUEL"].str.contains(name_contains, case=False, na=False)
        table = table[keep]
    return table.sort_values("n_valid", ascending=False)


def read_station(
    path: str | Path,
    station_id: str = STATION_ID,
    variables: Sequence[str] = ("T",),
    sep: str | None = None,
) -> pd.DataFrame:
    """Rows of ONE station in ONE archive.

    Args:
        path: archive (``.csv.gz``).
        station_id: NUM_POSTE of the station, e.g. "31069001".
        variables: columns to keep, e.g. ["T", "U"].
        sep: column separator; None = detected.

    Returns:
        A table with a ``time`` column (UTC) and one float column per variable. It is empty
        (zero rows) when the station is absent from the archive, for instance in a decade
        before the station opened.
    """
    variables = list(variables)
    pieces = [
        chunk[["time", *variables]]
        for chunk in _read_chunks(path, variables, sep, station_id)
    ]
    pieces = [piece for piece in pieces if not piece.empty]
    if not pieces:
        return pd.DataFrame(columns=["time", *variables])
    return pd.concat(pieces, ignore_index=True)


# =========================================================================================
# 4. Small cached extracts
# =========================================================================================


def _extract_name(archive_name: str, station_id: str, variables: Sequence[str]) -> str:
    """Name of the extract file of one archive, e.g. 'H_31_2010-2019__31069001__T.csv.gz'."""
    stem = _safe_file_name(archive_name).removesuffix(".csv.gz")
    clean = lambda text: re.sub(r"[^A-Za-z0-9_]", "_", text)
    return f"{stem}__{clean(station_id)}__{clean('-'.join(variables))}.csv.gz"


def _write_extract(frame: pd.DataFrame, path: Path) -> None:
    """Save an extract as a gzip CSV (written under a temporary name, then renamed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    try:
        frame.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_extract(path: str | Path) -> pd.DataFrame:
    """Read an extract written by extract_one_file: a ``time`` column (UTC) and the variables."""
    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame


def extract_one_file(
    url: str,
    archive_name: str,
    station_id: str,
    variables: Sequence[str],
    raw_dir: str | Path = "data/raw",
    interim_dir: str | Path = "data/interim",
    keep_raw: bool = False,
) -> Path:
    """Make (or find) the small extract of one archive.

    Steps: if the extract already exists, return its path (nothing is downloaded). Otherwise
    download the archive into ``raw_dir``, keep the rows of the station and the columns of the
    variables, save them in ``interim_dir``, then delete the archive unless ``keep_raw``.
    When the station is absent from the archive an EMPTY extract is saved, so the archive is
    not downloaded again next time.

    Args:
        url: https link of the archive.
        archive_name: file name of the archive, e.g. "H_31_2010-2019.csv.gz".
        station_id: NUM_POSTE of the station.
        variables: columns to keep, e.g. ["T"].
        raw_dir: folder of the archives (temporary).
        interim_dir: folder of the extracts (small, kept).
        keep_raw: True to keep the archive after extraction.

    Returns:
        The path of the extract file.
    """
    variables = list(variables)
    extract_path = Path(interim_dir) / _extract_name(
        archive_name, station_id, variables
    )
    if extract_path.exists():
        return extract_path
    archive = download_file(url, Path(raw_dir) / _safe_file_name(archive_name))
    _write_extract(read_station(archive, station_id, variables), extract_path)
    if not keep_raw:
        archive.unlink()  # the archive is big and we no longer need it
    return extract_path


# =========================================================================================
# 5. Build the hourly table
# =========================================================================================


def _assemble(
    pieces: Sequence[pd.DataFrame],
    station_id: str,
    variables: Sequence[str],
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Glue pieces (time + variables) into one regular hourly table, then check it.

    Shared by build_frame (pieces read from archives) and extract (pieces read from extracts).
    See build_frame for the meaning of the arguments and of the result.
    """
    variables = list(variables)
    pieces = [
        piece for piece in pieces if not piece.empty
    ]  # e.g. decades before the station
    if not pieces:
        raise ValueError(
            f"No row for station {station_id!r} in the given files. "
            "Run list_stations() on one file to see the station numbers it contains."
        )
    rows = pd.concat(pieces, ignore_index=True)  # pieces end to end
    # An hour present twice (two overlapping archives): keep the row of the later piece.
    rows = rows.drop_duplicates(subset="time", keep="last").sort_values("time")
    # time becomes the index; asfreq creates ONE row per hour, missing hours are NaN.
    frame = rows.set_index("time").asfreq(pd.offsets.Hour())
    frame = frame.loc[start:end]  # keep the wanted period (both bounds included)
    if frame.empty:
        raise ValueError(
            f"Station {station_id!r} has no row between {start!r} and {end!r}."
        )

    for variable in variables:
        if frame[variable].isna().all():
            raise ValueError(
                f"Station {station_id!r} has no value for {variable!r} between "
                f"{frame.index[0]} and {frame.index[-1]} (not measured there?)."
            )
        if variable in PLAUSIBLE_RANGES:
            low, high = PLAUSIBLE_RANGES[variable]
            if frame[variable].min() < low or frame[variable].max() > high:
                raise ValueError(
                    f"Implausible values for {variable!r} (min {frame[variable].min()}, "
                    f"max {frame[variable].max()}): check the unit and decimal mark."
                )
    return frame


def build_frame(
    paths: Sequence[str | Path],
    station_id: str = STATION_ID,
    variables: Sequence[str] = ("T",),
    start: str | None = None,
    end: str | None = None,
    sep: str | None = None,
) -> pd.DataFrame:
    """Glue several ARCHIVES into one hourly table of a single station.

    Args:
        paths: archives in chronological order. An hour present in two archives keeps the row
            of the later one.
        station_id: NUM_POSTE of the station.
        variables: column names of the source files, e.g. ["T", "TD", "U"].
        start, end: bounds in UTC as text ("1990", "2010-06", "2024-12-31 18:00"). Both bounds
            are included and a short text covers the whole period: end="2025" goes up to
            2025-12-31 23:00. None = as far as the archives go.
        sep: column separator; None = detected.

    Returns:
        A table indexed by a regular hourly UTC grid, one column per variable; a missing hour
        is NaN.

    Raises:
        ValueError: station absent, no data in the range, a variable entirely empty, or
            implausible values (wrong unit).
    """
    variables = list(variables)
    pieces = [read_station(p, station_id, variables, sep) for p in paths]
    return _assemble(pieces, station_id, variables, start, end)


def extract(
    station_id: str = STATION_ID,
    variables: Sequence[str] = ("T",),
    start: str | None = None,
    end: str | None = None,
    raw_dir: str | Path = "data/raw",
    interim_dir: str | Path = "data/interim",
    department: str = DEPARTMENT,
    files: pd.DataFrame | None = None,
    keep_raw: bool = False,
) -> pd.DataFrame:
    """Everything in one call: pick the archives, extract, glue.

    For each archive that overlaps [start, end]: make or find its small extract
    (extract_one_file), then glue the extracts (same rules as build_frame). Each selected
    archive must give an extract (else an error is raised), so a decade can never be missing
    silently: the result is continuous in time, with NaN where the station has no value.

    Args:
        station_id: NUM_POSTE of the station.
        variables: columns to keep, e.g. ["T", "U"].
        start, end: bounds in UTC (see build_frame); None = the longest period available.
        raw_dir: folder of the archives (temporary).
        interim_dir: folder of the extracts (small, reused by the next call).
        department: département number, used only when ``files`` is None.
        files: table of archives (list_files or files_from_urls); None = asked to data.gouv.fr.
        keep_raw: True to keep the archives after extraction.

    Returns:
        The hourly table, as build_frame.
    """
    variables = list(variables)
    if files is None:
        files = list_files(department)
    chosen = select_files(files, start, end)
    pieces = []
    for row in chosen.itertuples():
        path = extract_one_file(
            row.url, row.name, station_id, variables, raw_dir, interim_dir, keep_raw
        )
        pieces.append(_read_extract(path))
    return _assemble(pieces, station_id, variables, start, end)


# =========================================================================================
# 6. Check the gaps, cut into blocks
# =========================================================================================


def _longest_run(flags: np.ndarray) -> int:
    """Length of the longest run of consecutive True values.

    Example: [False, True, True, False, True] gives 2.
    """
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0  # extend the run or restart from zero
        longest = max(longest, current)
    return longest


def gap_report(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Summary of the missing hours of a regular hourly series or table.

    Args:
        data: a series, or a table with one column per variable (hourly index).

    Returns:
        One row per variable: first_valid and last_valid (first and last hour with a value),
        n_hours (length of the table), n_missing, share_missing (between 0 and 1) and
        longest_gap_hours (longest run of consecutive missing hours).
    """
    frame = data.to_frame() if isinstance(data, pd.Series) else data
    rows = {}
    for name in frame.columns:
        column = frame[name]
        missing = column.isna().to_numpy()
        rows[name] = {
            "first_valid": column.first_valid_index(),
            "last_valid": column.last_valid_index(),
            "n_hours": len(column),
            "n_missing": int(missing.sum()),
            "share_missing": float(missing.mean()),
            "longest_gap_hours": _longest_run(missing),
        }
    return pd.DataFrame(rows).T


def valid_per_year(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Number of non-missing values per calendar year (UTC), one column per variable.

    Use it to see how the observation rhythm changed over the decades: a full hourly year has
    8760 values (8784 in a leap year); a year observed every 3 hours has about 2920.

    Args:
        data: a series, or a table with one column per variable (datetime index).

    Returns:
        A table indexed by year with the number of valid values of each variable.
    """
    frame = data.to_frame() if isinstance(data, pd.Series) else data
    counts = frame.notna().groupby(frame.index.year).sum()
    counts.index.name = "year"
    return counts


def make_blocks(
    data: pd.Series | pd.DataFrame,
    starts: dict[str, str] | None = None,
    end: str | None = None,
) -> dict[str, pd.Series | pd.DataFrame]:
    """Cut a series (or table) into disjoint chronological blocks.

    Block i covers [starts[i], starts[i+1]); the last one ends at ``end`` (excluded). Nothing is
    shuffled and no block overlaps another, so a later block never leaks into an earlier one.

    Args:
        data: series or table with a UTC datetime index.
        starts: {block name: first hour}, in chronological order; None = BLOCK_STARTS.
        end: end of the last block (excluded); None = BLOCKS_END.

    Returns:
        {block name: the part of ``data`` inside the block}, in the order of ``starts``.

    Raises:
        ValueError: bounds not strictly increasing, or ``data`` does not cover the blocks
            (then add the missing archives).
    """
    starts = BLOCK_STARTS if starts is None else starts
    end = BLOCKS_END if end is None else end
    names = list(starts)
    bounds = [pd.Timestamp(starts[name], tz="UTC") for name in names]
    bounds.append(pd.Timestamp(end, tz="UTC"))
    # pairwise(a, b, c) gives (a, b), (b, c): each bound must be later than the previous one.
    if any(later <= earlier for earlier, later in itertools.pairwise(bounds)):
        raise ValueError(f"Block boundaries must be strictly increasing, got {bounds}")
    index = data.index
    if index.min() > bounds[0] or index.max() < bounds[-1] - pd.Timedelta(hours=1):
        raise ValueError(
            f"The data cover {index.min()} to {index.max()}, but the blocks need "
            f"{bounds[0]} to {bounds[-1]} (excluded). Add the missing files."
        )
    return {
        # & combines two True/False masks row by row: low <= time < high
        name: data[(index >= low) & (index < high)]
        for name, low, high in zip(names, bounds[:-1], bounds[1:])
    }
