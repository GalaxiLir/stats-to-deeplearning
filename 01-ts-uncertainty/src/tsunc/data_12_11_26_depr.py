"""Extract hourly Météo-France series: one station, a few variables, a time range.

Source: data.gouv.fr, dataset "Données climatologiques de base - horaires" (Licence Ouverte 2.0).
The dataset has one compressed CSV per département and period (about a decade); times are UTC.
All functions take explicit paths: nothing assumes where the repository lives.

One-call use (from the project folder)::

    frame = extract("31069001", variables=["T", "U"], start="1990", end="2025")

or step by step::

    files = list_files("31")                   # table of the files of a département
    files = select_files(files, "1990", "2025")  # keep the files that overlap the range
    paths = download_files(files, "data/raw")  # saved once, reused afterwards
    frame = build_frame(paths, "31069001", ["T", "U"], "1990", "2025")
    blocks = make_blocks(frame)                # train / val / calibration / test
"""

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
    "31069001"  # TOULOUSE-BLAGNAC (seen with list_stations on 2026-11-12); not 31070004
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
# 1. Which files exist, and which ones do we need?
# =========================================================================================


def _years_in_name(name: str) -> tuple[int, int] | None:
    """Find a period such as '2010-2019' in a file name (also 'previous-2020-2024')."""
    match = re.search(r"(\d{4})-(\d{4})", name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def files_from_urls(urls: dict[str, str]) -> pd.DataFrame:
    """Build the table of files from a hand-made {file name: link} dictionary.

    The name must contain the period, e.g. 'H_31_2010-2019.csv.gz'. Use this if list_files()
    cannot reach data.gouv.fr. Columns: name, url, first_year, last_year.
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
    """Ask data.gouv.fr for the hourly files of one département (one row per file).

    Needs an internet connection. The answer of the site was not checked against the real
    service when this code was written: if it fails, use files_from_urls() instead.
    """
    _require_https(api_url)
    request = urllib.request.Request(api_url, headers=_USER_AGENT)
    with urllib.request.urlopen(request, timeout=60) as response:
        dataset = json.load(response)
    resources = dataset.get("resources")
    if not isinstance(resources, list):
        raise TypeError(
            "Unexpected answer from data.gouv.fr (no list of 'resources'). "
            "Use files_from_urls() with links copied from the dataset page."
        )

    wanted = re.compile(rf"(^|[_-]){re.escape(department)}[_-]")
    urls = {}
    seen = []
    for resource in resources:
        url = str(resource.get("url") or "")
        url_name = Path(urllib.parse.urlparse(url).path).name
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
    """Keep the files whose period overlaps [start, end]. None means 'as far as possible'."""
    first = files["first_year"].min() if start is None else pd.Timestamp(start).year
    last = files["last_year"].max() if end is None else pd.Timestamp(end).year
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
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError(f"Only https links are accepted, got {url!r}")


def download_file(url: str, dest: str | Path, overwrite: bool = False) -> Path:
    """Download ``url`` to ``dest`` unless it is already there; return the path.

    The download goes to a temporary ``.part`` file and the result must be a gzip file, so
    an HTML error page is never kept as if it were data.
    """
    _require_https(url)
    dest = Path(dest)
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers=_USER_AGENT)
    try:
        with (
            urllib.request.urlopen(request, timeout=300) as response,
            open(tmp, "wb") as out,
        ):
            shutil.copyfileobj(response, out)
        with open(tmp, "rb") as handle:
            if handle.read(2) != b"\x1f\x8b":  # first two bytes of every gzip file
                raise RuntimeError(
                    f"{url} did not return a gzip file (an error page?). Open the link in a "
                    "browser to check that it still points to a data file."
                )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _safe_file_name(name: str) -> str:
    """Make a name coming from the internet safe to use as a file name."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if safe.startswith(".") or not safe.endswith(".csv.gz"):
        raise ValueError(f"Unexpected file name {name!r}")
    return safe


def download_files(files: pd.DataFrame, raw_dir: str | Path) -> list[Path]:
    """Download every file of the table into ``raw_dir``; return the paths in table order."""
    raw_dir = Path(raw_dir)
    return [
        download_file(row.url, raw_dir / _safe_file_name(row.name))
        for row in files.itertuples()
    ]


# =========================================================================================
# 3. Read the files
# =========================================================================================


def _detect_separator(path: str | Path) -> str:
    """Guess the column separator from the header line (';', tab or ',')."""
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
    """Read a file piece by piece; yield typed pieces (time in UTC, numbers as floats).

    Only the station columns, the time and the requested variables are read. If
    ``station_id`` is given, the other stations are dropped before any conversion.
    """
    sep = _detect_separator(path) if sep is None else sep
    wanted = [*_STATION_COLS, _TIME, *variables]
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
        usecols=wanted,
        dtype=str,  # read everything as text first: no silent guess about types
        chunksize=_CHUNKSIZE,
        encoding_errors="replace",
    )
    for chunk in reader:
        chunk["NUM_POSTE"] = chunk["NUM_POSTE"].fillna("").str.strip()
        if station_id is not None:
            chunk = chunk[chunk["NUM_POSTE"] == station_id]
        out = chunk[["NUM_POSTE", "NOM_USUEL"]].copy()
        out["NOM_USUEL"] = out["NOM_USUEL"].fillna("").str.strip()
        for column in ["LAT", "LON", "ALTI", *variables]:
            text = chunk[column].fillna("").str.replace(",", ".", regex=False)
            out[column] = pd.to_numeric(text, errors="coerce")  # unreadable -> NaN
        out["time"] = pd.to_datetime(
            chunk[_TIME], format="%Y%m%d%H", utc=True, errors="coerce"
        )
        yield out.dropna(subset=["time"])


def list_stations(
    path: str | Path,
    variable: str = "T",
    name_contains: str | None = None,
    sep: str | None = None,
) -> pd.DataFrame:
    """One row per station of a file: name, position, number of values of ``variable``.

    Use it on one file to check a station number. ``name_contains`` keeps the stations whose
    name matches a regular expression (not case-sensitive), e.g. "BLAGNAC|TOULOUSE".
    """
    pieces = []
    for chunk in _read_chunks(path, [variable], sep):
        valid = chunk.dropna(subset=[variable])
        grouped = valid.groupby("NUM_POSTE")["time"]
        pieces.append(
            pd.DataFrame(
                {
                    "NOM_USUEL": chunk.groupby("NUM_POSTE")["NOM_USUEL"].first(),
                    "LAT": chunk.groupby("NUM_POSTE")["LAT"].first(),
                    "LON": chunk.groupby("NUM_POSTE")["LON"].first(),
                    "ALTI": chunk.groupby("NUM_POSTE")["ALTI"].first(),
                    "first_time": grouped.min(),
                    "last_time": grouped.max(),
                    "n_valid": grouped.size(),
                }
            )
        )
    if not pieces:
        raise ValueError(f"{Path(path).name}: no readable row")
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
    """Rows of one station in one file: a ``time`` column (UTC) and one column per variable.

    The result is empty when the station is absent from the file (for instance a decade before
    it opened).
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


def build_frame(
    paths: Sequence[str | Path],
    station_id: str = STATION_ID,
    variables: Sequence[str] = ("T",),
    start: str | None = None,
    end: str | None = None,
    sep: str | None = None,
) -> pd.DataFrame:
    """Concatenate several files into one hourly table of a single station.

    - ``paths``: files in chronological order. A timestamp present in two files keeps the row
      of the later file.
    - ``variables``: column names of the source files, e.g. ["T", "TD", "U"].
    - ``start`` / ``end``: bounds in UTC, as text ("1990", "2010-06", "2024-12-31 18:00").
      Both bounds are included, and a short text covers the whole period: end="2025" goes up to
      2025-12-31 23:00. None means "as far as the files go".
    - Result: index = regular hourly UTC grid, one column per variable; a missing hour is NaN.
    """
    variables = list(variables)
    pieces = [read_station(p, station_id, variables, sep) for p in paths]
    pieces = [piece for piece in pieces if not piece.empty]
    if not pieces:
        raise ValueError(
            f"No row for station {station_id!r} in the given files. "
            "Run list_stations() on one file to see the station numbers it contains."
        )
    rows = pd.concat(pieces, ignore_index=True)
    rows = rows.drop_duplicates(subset="time", keep="last").sort_values("time")
    frame = rows.set_index("time").asfreq(pd.offsets.Hour())
    frame = frame.loc[start:end]
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


def extract(
    station_id: str = STATION_ID,
    variables: Sequence[str] = ("T",),
    start: str | None = None,
    end: str | None = None,
    raw_dir: str | Path = "data/raw",
    department: str = DEPARTMENT,
    files: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Everything in one call: pick the files, download them, build the table.

    ``files`` is the table of files (from list_files or files_from_urls); by default it is
    requested from data.gouv.fr. With start=None and end=None you get the longest period
    available in the files.
    """
    if files is None:
        files = list_files(department)
    chosen = select_files(files, start, end)
    paths = download_files(chosen, raw_dir)
    return build_frame(paths, station_id, variables, start, end)


# =========================================================================================
# 4. Check the gaps, cut into blocks
# =========================================================================================


def _longest_run(flags: np.ndarray) -> int:
    """Length of the longest run of consecutive True values."""
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def gap_report(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Summary of the missing hours: one row per variable."""
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


def make_blocks(
    data: pd.Series | pd.DataFrame,
    starts: dict[str, str] | None = None,
    end: str | None = None,
) -> dict[str, pd.Series | pd.DataFrame]:
    """Cut a series (or table) into disjoint chronological blocks.

    Default bounds: BLOCK_STARTS, BLOCKS_END. Block i covers [starts[i], starts[i+1]); the last
    one ends at ``end`` (excluded). Nothing is shuffled and no block overlaps another, so a
    later block never leaks into an earlier one.
    """
    starts = BLOCK_STARTS if starts is None else starts
    end = BLOCKS_END if end is None else end
    names = list(starts)
    bounds = [pd.Timestamp(starts[name], tz="UTC") for name in names]
    bounds.append(pd.Timestamp(end, tz="UTC"))
    if any(later <= earlier for earlier, later in itertools.pairwise(bounds)):
        raise ValueError(f"Block boundaries must be strictly increasing, got {bounds}")
    index = data.index
    if index.min() > bounds[0] or index.max() < bounds[-1] - pd.Timedelta(hours=1):
        raise ValueError(
            f"The data cover {index.min()} to {index.max()}, but the blocks need "
            f"{bounds[0]} to {bounds[-1]} (excluded). Add the missing files."
        )
    return {
        name: data[(index >= low) & (index < high)]
        for name, low, high in zip(names, bounds[:-1], bounds[1:])
    }
