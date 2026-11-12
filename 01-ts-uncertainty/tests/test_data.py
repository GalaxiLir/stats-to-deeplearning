"""Tests of tsunc.data.

How to read this file
---------------------
A test is a function whose name starts with ``test_``. pytest calls each one; a test passes when
no ``assert`` fails and no error is raised (``testthat::expect_equal`` in R).
Every test below follows the same pattern: build a tiny input (a few hours of fake data), call
one function, compare the result with the value we know it must give.
No test touches the internet: downloads are replaced by a fake (``monkeypatch``).

Helpers (not tests): _rows, _write, _gz_bytes, _hourly, _fake_internet.
Sections: reading archives, building the table, extracts, gaps, blocks, file table, download.
"""

import gzip
import io
import itertools
import json

import numpy as np
import pandas as pd
import pytest

from tsunc import data

# Two fake station numbers. BLAGNAC plays the role of the station we want, OTHER the others.
BLAGNAC, OTHER = "31069001", "31555005"


# ------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------


def _rows(poste, name, start, temperatures, humidity=None):
    """Make fake hourly rows for one station, as a list of dictionaries (one per row).

    Args:
        poste: station number (NUM_POSTE).
        name: station name (NOM_USUEL).
        start: first hour, e.g. "2024-01-01" (UTC).
        temperatures: one value per hour; None gives an empty field (missing value).
        humidity: one value per hour for the column U; None = 50 everywhere.

    The other columns (RR1, QT, TD) are there to look like the real files, which have ~190.
    """
    hours = pd.date_range(
        start, periods=len(temperatures), freq=pd.offsets.Hour(), tz="UTC"
    )
    humidity = [50.0] * len(temperatures) if humidity is None else humidity
    return [
        {
            "NUM_POSTE": poste,
            "NOM_USUEL": name,
            "LAT": 43.62,
            "LON": 1.38,
            "ALTI": 151,
            "AAAAMMJJHH": hour.strftime("%Y%m%d%H"),  # 2024-01-01 05:00 -> "2024010105"
            "RR1": 0.0,
            "T": temp,
            "QT": 1,
            "TD": 5.0,
            "U": hum,
        }
        for hour, temp, hum in zip(hours, temperatures, humidity)
    ]


def _write(path, rows, sep=";", decimal="."):
    """Write fake rows to a gzip CSV, like an archive; return its path.

    ``sep`` and ``decimal`` let a test imitate other file layouts (comma, tab, decimal comma).
    """
    pd.DataFrame(rows).to_csv(
        path, sep=sep, index=False, decimal=decimal, compression="gzip"
    )
    return path


def _gz_bytes(rows):
    """The content of a gzip CSV as bytes: what a fake server would send."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as handle:
        handle.write(pd.DataFrame(rows).to_csv(sep=";", index=False).encode())
    return buffer.getvalue()


def _fake_internet(monkeypatch, payloads):
    """Replace the real download by a fake server; return the list of links requested.

    Args:
        payloads: {link: bytes} what the fake server answers for each link.

    The returned list lets a test check which links were really asked, hence whether a second
    call used the cache instead of downloading again.
    """
    asked = []

    def fake_urlopen(request, timeout=None):
        asked.append(request.full_url)
        return io.BytesIO(payloads[request.full_url])

    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    return asked


def _hourly(start, end):
    """A hourly series 0, 1, 2, ... between two dates (UTC), named "T"."""
    index = pd.date_range(start, end, freq=pd.offsets.Hour(), tz="UTC")
    return pd.Series(np.arange(len(index), dtype=float), index=index, name="T")


# ------------------------------------------------------------------------------------------
# 1. Reading archives
# ------------------------------------------------------------------------------------------


def test_read_station_keeps_one_station_and_parses_time_and_values(tmp_path):
    """read_station keeps only the asked station and turns text into dates and numbers.

    Scenario: a file with 3 rows for BLAGNAC (one empty temperature) and 2 rows for OTHER.
    Call:     read_station(path, BLAGNAC, ["T", "U"])
    Expected: 3 rows, columns time/T/U, first time = 2024-01-01 00:00 UTC, one NaN in T.
    """
    rows = _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [10.5, None, 12.0])
    rows += _rows(OTHER, "AUTRE", "2024-01-01", [5.0, 6.0])
    path = _write(tmp_path / "a.csv.gz", rows)

    result = data.read_station(path, BLAGNAC, ["T", "U"])

    assert list(result.columns) == ["time", "T", "U"]
    assert len(result) == 3
    assert result["time"].iloc[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert result["T"].iloc[0] == pytest.approx(10.5)
    assert result["T"].isna().sum() == 1
    assert (result["U"] == 50.0).all()


def test_decimal_comma_is_accepted(tmp_path):
    """A decimal comma ("10,5") is read as 10.5.

    Scenario: a file written with decimal=",".
    Expected: the values come out as the floats 10.5 and 11.25.
    """
    path = _write(
        tmp_path / "a.csv.gz",
        _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [10.5, 11.25]),
        decimal=",",
    )
    assert data.read_station(path, BLAGNAC)["T"].tolist() == [10.5, 11.25]


@pytest.mark.parametrize("sep", [";", ",", "\t"])
def test_separator_is_detected_automatically(tmp_path, sep):
    """The column separator (;  ,  or tab) is found without being told.

    ``parametrize`` runs this one function three times, once per separator: this is why the
    total number of tests is larger than the number of ``def test_`` lines.
    """
    path = _write(
        tmp_path / "a.csv.gz",
        _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [1.0, 2.0]),
        sep=sep,
    )
    assert data.read_station(path, BLAGNAC)["T"].tolist() == [1.0, 2.0]


def test_missing_columns_give_a_helpful_error(tmp_path):
    """A wrong variable name or a wrong file stops with a message that says what was found.

    Scenario 1: ask for a column "NOPE" that is not in the file.
    Scenario 2: a file without the expected columns at all.
    Expected:   ValueError naming the missing column / the first columns found.
    """
    path = _write(tmp_path / "a.csv.gz", _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [1.0]))
    with pytest.raises(ValueError, match="missing columns .*NOPE"):
        data.read_station(path, BLAGNAC, ["T", "NOPE"])

    bad = pd.DataFrame({"a": [1], "b": [2]})
    bad.to_csv(tmp_path / "b.csv.gz", sep=";", index=False, compression="gzip")
    with pytest.raises(ValueError, match="First columns found"):
        data.read_station(tmp_path / "b.csv.gz", BLAGNAC)


def test_list_stations_counts_valid_values_and_filters_by_name(tmp_path):
    """list_stations gives one row per station, with the number of non-empty temperatures.

    Scenario: BLAGNAC has 3 rows of which one empty; OTHER has 2 rows.
    Call:     list_stations(path), then list_stations(path, name_contains="blag").
    Expected: n_valid = 2 for BLAGNAC; first/last time = first/last hour WITH a value;
              the name filter (not case-sensitive) keeps only BLAGNAC.
    """
    rows = _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [10.5, None, 12.0])
    rows += _rows(OTHER, "AUTRE", "2024-01-01", [5.0, 6.0])
    path = _write(tmp_path / "a.csv.gz", rows)

    stations = data.list_stations(path)
    assert set(stations.index) == {BLAGNAC, OTHER}
    assert stations.loc[BLAGNAC, "n_valid"] == 2
    assert stations.loc[BLAGNAC, "NOM_USUEL"] == "BLAGNAC"
    assert stations.loc[BLAGNAC, "first_time"] == pd.Timestamp(
        "2024-01-01 00:00", tz="UTC"
    )
    assert stations.loc[BLAGNAC, "last_time"] == pd.Timestamp(
        "2024-01-01 02:00", tz="UTC"
    )
    assert list(data.list_stations(path, name_contains="blag").index) == [BLAGNAC]


# ------------------------------------------------------------------------------------------
# 2. Building the hourly table from archives
# ------------------------------------------------------------------------------------------


def _two_overlapping_files(tmp_path):
    """Two archives for BLAGNAC that overlap on 2024-01-02 and contain two gaps.

    File a: 2024-01-01 and 2024-01-02 (48 h, temperature 12), empty value at 2024-01-01 05:00.
    File b: 2024-01-02 to 2024-01-04 (72 h, temperature 20), no row at all at 2024-01-03 10:00.
    """
    first = _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [12.0] * 48)
    first[5]["T"] = None
    second = _rows(BLAGNAC, "BLAGNAC", "2024-01-02", [20.0] * 72)
    del second[34]
    return [
        _write(tmp_path / "a.csv.gz", first),
        _write(tmp_path / "b.csv.gz", second),
    ]


def test_build_frame_concatenates_overlapping_files_and_marks_gaps(tmp_path):
    """build_frame glues archives on a regular hourly grid; gaps become NaN.

    Call:     build_frame([a, b], BLAGNAC)
    Expected: 96 rows (2024-01-01 00:00 to 2024-01-04 23:00), exactly one hour between rows,
              the overlapping day takes the value of the LATER file (20), and the two gaps
              (one empty value, one absent row) are both NaN: gap_report counts 2, longest 1.
    """
    paths = _two_overlapping_files(tmp_path)

    frame = data.build_frame(paths, BLAGNAC)

    steps = frame.index.to_series().diff().dropna()
    assert (steps == pd.Timedelta(hours=1)).all()  # regular hourly grid
    assert len(frame) == 96
    assert frame.loc[pd.Timestamp("2024-01-01 23:00", tz="UTC"), "T"] == 12.0
    assert (
        frame.loc[pd.Timestamp("2024-01-02 00:00", tz="UTC"), "T"] == 20.0
    )  # later wins
    assert frame["T"].isna().sum() == 2
    report = data.gap_report(frame)
    assert report.loc["T", "n_missing"] == 2
    assert report.loc["T", "longest_gap_hours"] == 1


def test_build_frame_start_and_end_are_inclusive_and_accept_short_text(tmp_path):
    """start/end cut the period; both bounds are included; "2024" means the whole year.

    Call 1:   start="2024-01-02", end="2024-01-02" -> the 24 hours of that day.
    Call 2:   start="2000", end="2024" -> everything (the whole year 2024 is included).
    Call 3:   start="2030" -> error, there is no data after the files.
    """
    paths = _two_overlapping_files(tmp_path)

    day = data.build_frame(paths, BLAGNAC, start="2024-01-02", end="2024-01-02")
    assert len(day) == 24
    assert day.index[0] == pd.Timestamp("2024-01-02 00:00", tz="UTC")
    assert day.index[-1] == pd.Timestamp("2024-01-02 23:00", tz="UTC")

    year = data.build_frame(paths, BLAGNAC, start="2000", end="2024")
    assert len(year) == 96

    with pytest.raises(ValueError, match="no row between"):
        data.build_frame(paths, BLAGNAC, start="2030")


def test_build_frame_several_variables(tmp_path):
    """Several variables give several columns, in the order asked.

    Call:     build_frame([path], BLAGNAC, ["T", "U"])
    Expected: columns ["T", "U"]; U holds the humidity values 60 and 70.
    """
    path = _write(
        tmp_path / "a.csv.gz",
        _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [1.0, 2.0], humidity=[60.0, 70.0]),
    )
    frame = data.build_frame([path], BLAGNAC, ["T", "U"])
    assert list(frame.columns) == ["T", "U"]
    assert frame["U"].tolist() == [60.0, 70.0]


def test_build_frame_refuses_unknown_station_empty_variable_and_bad_units(tmp_path):
    """Three situations where the code must stop instead of returning doubtful data.

    1. Unknown station -> error that points to list_stations.
    2. A variable that is empty everywhere (not measured) -> error naming it.
    3. Temperatures of 105 and 215 (tenths of degrees taken for degrees) -> "Implausible".
    """
    path = _write(
        tmp_path / "a.csv.gz", _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [1.0, 2.0])
    )
    with pytest.raises(ValueError, match="list_stations"):
        data.build_frame([path], "99999999")

    empty = _write(
        tmp_path / "b.csv.gz", _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [None, None])
    )
    with pytest.raises(ValueError, match="no value for 'T'"):
        data.build_frame([empty], BLAGNAC)

    tenths = _write(
        tmp_path / "c.csv.gz", _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [105.0, 215.0])
    )
    with pytest.raises(ValueError, match="Implausible"):
        data.build_frame([tenths], BLAGNAC)


# ------------------------------------------------------------------------------------------
# 3. Small extracts (download, keep the station, delete the archive)
# ------------------------------------------------------------------------------------------


def test_extract_one_file_saves_an_extract_deletes_the_archive_and_reuses_it(
    tmp_path, monkeypatch
):
    """One archive -> one small extract; the archive disappears; a second call downloads nothing.

    Scenario: a fake server holds an archive with rows of BLAGNAC and of OTHER.
    Call 1:   extract_one_file(...) -> downloads, keeps only BLAGNAC / column T, deletes archive.
    Call 2:   same call -> returns the same extract, the server is not asked again.
    """
    rows = _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [1.0, 2.0])
    rows += _rows(OTHER, "AUTRE", "2024-01-01", [9.0, 9.0, 9.0])
    asked = _fake_internet(monkeypatch, {"https://example.org/a": _gz_bytes(rows)})
    raw, interim = tmp_path / "raw", tmp_path / "interim"

    path = data.extract_one_file(
        "https://example.org/a", "H_31_2020-2024.csv.gz", BLAGNAC, ["T"], raw, interim
    )

    assert path.name == f"H_31_2020-2024__{BLAGNAC}__T.csv.gz"
    assert list(raw.glob("*")) == []  # the big archive was deleted
    assert data._read_extract(path)["T"].tolist() == [1.0, 2.0]  # only BLAGNAC kept

    again = data.extract_one_file(
        "https://example.org/a", "H_31_2020-2024.csv.gz", BLAGNAC, ["T"], raw, interim
    )
    assert again == path
    assert len(asked) == 1  # second call used the extract


def test_extract_one_file_keep_raw_and_station_absent(tmp_path, monkeypatch):
    """keep_raw=True keeps the archive; an absent station gives an EMPTY extract (also cached).

    Scenario: an archive that contains only OTHER (the station did not exist in that decade).
    Expected: an empty extract is saved, the archive is kept (keep_raw=True), and a second call
              does not download again.
    """
    rows = _rows(OTHER, "AUTRE", "2024-01-01", [9.0, 9.0])
    asked = _fake_internet(monkeypatch, {"https://example.org/a": _gz_bytes(rows)})
    raw, interim = tmp_path / "raw", tmp_path / "interim"
    args = ("https://example.org/a", "H_31_1900-1909.csv.gz", BLAGNAC, ["T"])

    path = data.extract_one_file(*args, raw, interim, keep_raw=True)

    assert data._read_extract(path).empty
    assert (raw / "H_31_1900-1909.csv.gz").exists()
    data.extract_one_file(*args, raw, interim, keep_raw=True)
    assert len(asked) == 1


def test_extract_end_to_end_with_a_fake_server(tmp_path, monkeypatch):
    """extract(): choose archives, make extracts, glue them; the second call is offline.

    Scenario: an old archive (2009-12-31 22:00 to 2010-01-01 01:00, values 1 2 3 4) and a new one
              (from 2010-01-01 00:00, values 3.5 4.5 5.5) that overlap by two hours.
    Call:     extract(BLAGNAC, ["T"], start="2009", files=...)
    Expected: [1, 2, 3.5, 4.5, 5.5] (the later archive wins on the overlap), no archive left in
              raw/, one extract per archive in interim/, and no new download on a second call.
    """
    old = _rows(BLAGNAC, "BLAGNAC", "2009-12-31 22:00", [1.0, 2.0, 3.0, 4.0])
    new = _rows(BLAGNAC, "BLAGNAC", "2010-01-01 00:00", [3.5, 4.5, 5.5])
    asked = _fake_internet(
        monkeypatch,
        {
            "https://example.org/old": _gz_bytes(old),
            "https://example.org/new": _gz_bytes(new),
        },
    )
    files = data.files_from_urls(
        {
            "H_31_2000-2009.csv.gz": "https://example.org/old",
            "H_31_2010-2019.csv.gz": "https://example.org/new",
        }
    )
    raw, interim = tmp_path / "raw", tmp_path / "interim"

    frame = data.extract(
        BLAGNAC, ["T"], start="2009", files=files, raw_dir=raw, interim_dir=interim
    )

    assert asked == ["https://example.org/old", "https://example.org/new"]
    assert frame["T"].tolist() == [1.0, 2.0, 3.5, 4.5, 5.5]
    assert list(raw.glob("*.csv.gz")) == []
    assert len(list(interim.glob("*.csv.gz"))) == 2

    again = data.extract(
        BLAGNAC, ["T"], start="2009", files=files, raw_dir=raw, interim_dir=interim
    )
    assert len(asked) == 2  # nothing downloaded the second time
    pd.testing.assert_frame_equal(again, frame)


def test_extract_only_downloads_the_archives_overlapping_the_range(
    tmp_path, monkeypatch
):
    """With start="2010" the archive of 2000-2009 is not even downloaded."""
    new = _rows(BLAGNAC, "BLAGNAC", "2010-01-01 00:00", [3.5, 4.5])
    asked = _fake_internet(monkeypatch, {"https://example.org/new": _gz_bytes(new)})
    files = data.files_from_urls(
        {
            "H_31_2000-2009.csv.gz": "https://example.org/old",
            "H_31_2010-2019.csv.gz": "https://example.org/new",
        }
    )
    frame = data.extract(
        BLAGNAC,
        ["T"],
        start="2010",
        files=files,
        raw_dir=tmp_path / "r",
        interim_dir=tmp_path / "i",
    )
    assert asked == ["https://example.org/new"]
    assert frame["T"].tolist() == [3.5, 4.5]


# ------------------------------------------------------------------------------------------
# 4. Gaps and yearly counts
# ------------------------------------------------------------------------------------------


def test_gap_report_longest_run_for_series_and_frame():
    """gap_report counts missing hours and the longest run of consecutive missing hours.

    Scenario: 10 hours with NaN at positions 1,2,3 (a run of 3), 5, 8 and 9 (a run of 2).
    Expected: 6 missing, longest gap 3, first valid = position 0, last valid = position 7.
    """
    index = pd.date_range("2024-01-01", periods=10, freq=pd.offsets.Hour(), tz="UTC")
    values = [1.0, np.nan, np.nan, np.nan, 2.0, np.nan, 3.0, 4.0, np.nan, np.nan]
    report = data.gap_report(pd.Series(values, index=index, name="T"))
    assert report.loc["T", "n_missing"] == 6
    assert report.loc["T", "longest_gap_hours"] == 3
    assert report.loc["T", "first_valid"] == index[0]
    assert report.loc["T", "last_valid"] == index[7]


def test_valid_per_year_shows_a_change_of_observation_rhythm():
    """valid_per_year counts values per year: it reveals a switch from 3-hourly to hourly data.

    Scenario: year 2019 observed every 3 hours (365 * 8 = 2920 values), year 2020 every hour
              (leap year: 366 * 24 = 8784 values).
    Expected: counts 2920 and 8784.
    """
    series = _hourly("2019-01-01 00:00", "2020-12-31 23:00")
    keep = (series.index.year == 2020) | (series.index.hour % 3 == 0)
    series[~keep] = np.nan
    counts = data.valid_per_year(series)
    assert counts.loc[2019, "T"] == 2920
    assert counts.loc[2020, "T"] == 8784


# ------------------------------------------------------------------------------------------
# 5. Chronological blocks
# ------------------------------------------------------------------------------------------


def test_make_blocks_are_chronological_disjoint_and_cover_the_period():
    """The four blocks follow each other in time, never overlap, and lose no hour.

    Scenario: a series from 2009-12-31 to 2024-01-02.
    Expected: names train/val/calibration/test; each block ends before the next one starts
              (no leakage of the future); test = 2023 = 365 * 24 hours; the total number of
              hours equals the hours of 2010-01-01 to 2023-12-31.
    """
    series = _hourly("2009-12-31 00:00", "2024-01-02 23:00")
    blocks = data.make_blocks(series)

    names = list(blocks)
    assert names == ["train", "val", "calibration", "test"]
    for earlier, later in itertools.pairwise(names):
        assert blocks[earlier].index.max() < blocks[later].index.min()
    assert len(blocks["test"]) == 365 * 24
    assert blocks["train"].index.min() == pd.Timestamp("2010-01-01", tz="UTC")
    assert sum(len(b) for b in blocks.values()) == len(
        series.loc["2010-01-01":"2023-12-31 23:00"]
    )


def test_make_blocks_accepts_a_table():
    """make_blocks also cuts a table with several variables (here T and U)."""
    series = _hourly("2010-01-01 00:00", "2024-01-01 00:00")
    table = pd.DataFrame({"T": series, "U": series * 2})
    blocks = data.make_blocks(table)
    assert isinstance(blocks["test"], pd.DataFrame)
    assert list(blocks["test"].columns) == ["T", "U"]
    assert len(blocks["test"]) == 365 * 24


def test_make_blocks_rejects_bad_boundaries_and_short_series():
    """Bad block bounds, or data that do not cover the blocks, stop with a clear error.

    1. Bounds that go backwards (2010 then 2009) -> "increasing".
    2. Data starting in 2012 while the first block starts in 2010 -> "Add the missing files".
    3. Data ending in mid-2023 while the last block ends 2024-01-01 -> same message.
    """
    series = _hourly("2010-01-01 00:00", "2024-01-01 00:00")
    with pytest.raises(ValueError, match="increasing"):
        data.make_blocks(
            series,
            starts={"train": "2010-01-01", "val": "2009-01-01"},
            end="2024-01-01",
        )
    with pytest.raises(ValueError, match="Add the missing files"):
        data.make_blocks(_hourly("2012-01-01 00:00", "2024-01-01 00:00"))
    with pytest.raises(ValueError, match="Add the missing files"):
        data.make_blocks(_hourly("2010-01-01 00:00", "2023-06-01 00:00"))


# ------------------------------------------------------------------------------------------
# 6. Table of archives (names, periods) and download
# ------------------------------------------------------------------------------------------

URLS = {
    "H_31_2010-2019.csv.gz": "https://example.org/b",
    "H_31_previous-2020-2024.csv.gz": "https://example.org/c",
    "H_31_2000-2009.csv.gz": "https://example.org/a",
    "H_31_latest-2025-2026.csv.gz": "https://example.org/d",
}


def test_files_from_urls_sorts_by_period_and_needs_a_period():
    """files_from_urls reads the period in each name and sorts from oldest to newest.

    Scenario: four names given in disorder, one of them with "previous-" and one with "latest-".
    Expected: first_year 2000, 2010, 2020, 2025; a name without period raises ValueError.
    """
    files = data.files_from_urls(URLS)
    assert files["first_year"].tolist() == [2000, 2010, 2020, 2025]
    assert files["last_year"].tolist() == [2009, 2019, 2024, 2026]
    with pytest.raises(ValueError, match="No period"):
        data.files_from_urls({"H_31.csv.gz": "https://example.org/x"})


def test_select_files_keeps_the_files_overlapping_the_range():
    """select_files keeps the archives whose years overlap the requested range.

    ("2015", "2021") -> 2010-2019 and 2020-2024; no bounds -> all four; a range before the first
    archive -> error.
    """
    files = data.files_from_urls(URLS)
    assert data.select_files(files, "2015", "2021")["first_year"].tolist() == [
        2010,
        2020,
    ]
    assert len(data.select_files(files)) == 4
    assert data.select_files(files, "2024-12-31", None)["first_year"].tolist() == [
        2020,
        2025,
    ]
    with pytest.raises(ValueError, match="No file covers"):
        data.select_files(files, "1900", "1950")


def test_list_files_reads_the_dataset_description(monkeypatch):
    """list_files reads the JSON description of the dataset and keeps the right département.

    Scenario: a fake answer with archives of département 31, one of 32, one of 310 (must not be
    confused with 31), a PDF, and one archive whose name is only in its link.
    Expected: three archives for "31" (2000, 2010, 2025); asking "75" raises ValueError.
    """
    resources = [
        {"title": "H_31_2010-2019.csv.gz", "url": "https://example.org/a"},
        {"title": "H_31_latest-2025-2026.csv.gz", "url": "https://example.org/b"},
        {"title": "H_32_2010-2019.csv.gz", "url": "https://example.org/other-dept"},
        {"title": "H_310_2010-2019.csv.gz", "url": "https://example.org/dept-310"},
        {"title": "Description des champs", "url": "https://example.org/doc.pdf"},
        {"title": "Fichier", "url": "https://example.org/dl/H_31_2000-2009.csv.gz"},
    ]
    body = json.dumps({"resources": resources}).encode()
    monkeypatch.setattr(
        data.urllib.request, "urlopen", lambda request, timeout=None: io.BytesIO(body)
    )

    files = data.list_files("31")

    assert files["first_year"].tolist() == [2000, 2010, 2025]
    assert (
        files["name"].tolist()[0] == "H_31_2000-2009.csv.gz"
    )  # name taken from the link
    with pytest.raises(ValueError, match="No .csv.gz file"):
        data.list_files("75")


def test_list_files_explains_an_unexpected_answer(monkeypatch):
    """If the site answers with another structure, the error tells to use files_from_urls."""
    body = json.dumps({"resources": {"href": "..."}}).encode()
    monkeypatch.setattr(
        data.urllib.request, "urlopen", lambda request, timeout=None: io.BytesIO(body)
    )
    with pytest.raises(TypeError, match="files_from_urls"):
        data.list_files("31")


def test_download_file_saves_then_uses_the_cache(tmp_path, monkeypatch):
    """download_file saves the file once; a second call with the same destination does nothing.

    Expected: the file holds exactly what the fake server sent; the server was asked once.
    """
    payload = gzip.compress(b"a;b\n1;2\n")
    asked = _fake_internet(monkeypatch, {"https://example.org/f": payload})
    dest = tmp_path / "raw" / "f.csv.gz"  # the folder "raw" does not exist yet

    assert data.download_file("https://example.org/f", dest) == dest
    assert dest.read_bytes() == payload
    data.download_file("https://example.org/f", dest)
    assert len(asked) == 1


def test_download_file_refuses_html_pages_and_non_https_links(tmp_path, monkeypatch):
    """Safety checks of download_file.

    1. The server answers with an HTML page instead of a gzip file -> RuntimeError, nothing kept,
       not even the temporary .part file.
    2. http:// and file:// links are refused (only https).
    """
    monkeypatch.setattr(
        data.urllib.request,
        "urlopen",
        lambda request, timeout=None: io.BytesIO(b"<html>oops"),
    )
    dest = tmp_path / "f.csv.gz"

    with pytest.raises(RuntimeError, match="gzip"):
        data.download_file("https://example.org/f", dest)
    assert not dest.exists()
    assert not dest.with_name("f.csv.gz.part").exists()

    with pytest.raises(ValueError, match="https"):
        data.download_file("http://example.org/f", dest)
    with pytest.raises(ValueError, match="https"):
        data.download_file("file:///etc/passwd", dest)


def test_download_files_uses_safe_names(tmp_path, monkeypatch):
    """File names coming from the internet are neutralised before being used on disk.

    1. A normal name is kept as is.
    2. A "/" in a name becomes "_": the file stays inside the chosen folder.
    3. A name starting with "../" or not ending with ".csv.gz" is refused.
    """
    payload = gzip.compress(b"a;b\n")
    monkeypatch.setattr(
        data.urllib.request,
        "urlopen",
        lambda request, timeout=None: io.BytesIO(payload),
    )
    good = data.files_from_urls({"H_31_2010-2019.csv.gz": "https://example.org/a"})
    assert [p.name for p in data.download_files(good, tmp_path)] == [
        "H_31_2010-2019.csv.gz"
    ]

    nested = data.files_from_urls({"sub/dir_2010-2019.csv.gz": "https://example.org/a"})
    assert data.download_files(nested, tmp_path)[0].parent == tmp_path
    evil = data.files_from_urls({"../x_2010-2019.csv.gz": "https://example.org/a"})
    with pytest.raises(ValueError, match="Unexpected file name"):
        data.download_files(evil, tmp_path)
    bad = data.files_from_urls({".hidden_2010-2019.txt": "https://example.org/a"})
    with pytest.raises(ValueError, match="Unexpected file name"):
        data.download_files(bad, tmp_path)
