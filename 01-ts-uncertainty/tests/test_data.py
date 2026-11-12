import gzip
import io
import itertools
import json

import numpy as np
import pandas as pd
import pytest

from tsunc import data

BLAGNAC, OTHER = "31069001", "31555005"


def _rows(poste, name, start, temperatures, humidity=None):
    """Hourly rows from `start`; None gives an empty field. Extra columns (RR1...) are included."""
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
            "AAAAMMJJHH": hour.strftime("%Y%m%d%H"),
            "RR1": 0.0,
            "T": temp,
            "QT": 1,
            "TD": 5.0,
            "U": hum,
        }
        for hour, temp, hum in zip(hours, temperatures, humidity)
    ]


def _write(path, rows, sep=";", decimal="."):
    """Write a small gzip CSV with the layout of the source files."""
    pd.DataFrame(rows).to_csv(
        path, sep=sep, index=False, decimal=decimal, compression="gzip"
    )
    return path


def _gz_bytes(rows):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as handle:
        handle.write(pd.DataFrame(rows).to_csv(sep=";", index=False).encode())
    return buffer.getvalue()


# --- Reading ------------------------------------------------------------------------------


def test_read_station_keeps_one_station_and_parses_time_and_values(tmp_path):
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
    path = _write(
        tmp_path / "a.csv.gz",
        _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [10.5, 11.25]),
        decimal=",",
    )
    assert data.read_station(path, BLAGNAC)["T"].tolist() == [10.5, 11.25]


@pytest.mark.parametrize("sep", [";", ",", "\t"])
def test_separator_is_detected_automatically(tmp_path, sep):
    path = _write(
        tmp_path / "a.csv.gz",
        _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [1.0, 2.0]),
        sep=sep,
    )
    assert data.read_station(path, BLAGNAC)["T"].tolist() == [1.0, 2.0]


def test_missing_columns_give_a_helpful_error(tmp_path):
    path = _write(tmp_path / "a.csv.gz", _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [1.0]))
    with pytest.raises(ValueError, match="missing columns .*NOPE"):
        data.read_station(path, BLAGNAC, ["T", "NOPE"])

    bad = pd.DataFrame({"a": [1], "b": [2]})
    bad.to_csv(tmp_path / "b.csv.gz", sep=";", index=False, compression="gzip")
    with pytest.raises(ValueError, match="First columns found"):
        data.read_station(tmp_path / "b.csv.gz", BLAGNAC)


def test_list_stations_counts_valid_values_and_filters_by_name(tmp_path):
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


# --- Building the table ---------------------------------------------------------------------


def _two_overlapping_files(tmp_path):
    first = _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [12.0] * 48)
    first[5]["T"] = None  # empty temperature at 2024-01-01 05:00
    second = _rows(BLAGNAC, "BLAGNAC", "2024-01-02", [20.0] * 72)
    del second[34]  # no row at all for 2024-01-03 10:00
    return [
        _write(tmp_path / "a.csv.gz", first),
        _write(tmp_path / "b.csv.gz", second),
    ]


def test_build_frame_concatenates_overlapping_files_and_marks_gaps(tmp_path):
    paths = _two_overlapping_files(tmp_path)

    frame = data.build_frame(paths, BLAGNAC)

    steps = frame.index.to_series().diff().dropna()
    assert (steps == pd.Timedelta(hours=1)).all()  # regular hourly grid
    assert len(frame) == 96  # 2024-01-01 00:00 to 2024-01-04 23:00
    assert frame.loc[pd.Timestamp("2024-01-01 23:00", tz="UTC"), "T"] == 12.0
    assert (
        frame.loc[pd.Timestamp("2024-01-02 00:00", tz="UTC"), "T"] == 20.0
    )  # later wins
    assert frame["T"].isna().sum() == 2
    report = data.gap_report(frame)
    assert report.loc["T", "n_missing"] == 2
    assert report.loc["T", "longest_gap_hours"] == 1


def test_build_frame_start_and_end_are_inclusive_and_accept_short_text(tmp_path):
    paths = _two_overlapping_files(tmp_path)

    day = data.build_frame(paths, BLAGNAC, start="2024-01-02", end="2024-01-02")
    assert len(day) == 24
    assert day.index[0] == pd.Timestamp("2024-01-02 00:00", tz="UTC")
    assert day.index[-1] == pd.Timestamp("2024-01-02 23:00", tz="UTC")

    year = data.build_frame(paths, BLAGNAC, start="2000", end="2024")
    assert len(year) == 96  # the whole year 2024 is included, nothing before the data

    with pytest.raises(ValueError, match="no row between"):
        data.build_frame(paths, BLAGNAC, start="2030")


def test_build_frame_several_variables(tmp_path):
    path = _write(
        tmp_path / "a.csv.gz",
        _rows(BLAGNAC, "BLAGNAC", "2024-01-01", [1.0, 2.0], humidity=[60.0, 70.0]),
    )
    frame = data.build_frame([path], BLAGNAC, ["T", "U"])
    assert list(frame.columns) == ["T", "U"]
    assert frame["U"].tolist() == [60.0, 70.0]


def test_build_frame_refuses_unknown_station_empty_variable_and_bad_units(tmp_path):
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


def test_gap_report_longest_run_for_series_and_frame():
    index = pd.date_range("2024-01-01", periods=10, freq=pd.offsets.Hour(), tz="UTC")
    values = [1.0, np.nan, np.nan, np.nan, 2.0, np.nan, 3.0, 4.0, np.nan, np.nan]
    report = data.gap_report(pd.Series(values, index=index, name="T"))
    assert report.loc["T", "n_missing"] == 6
    assert report.loc["T", "longest_gap_hours"] == 3
    assert report.loc["T", "first_valid"] == index[0]
    assert report.loc["T", "last_valid"] == index[7]


# --- Blocks ---------------------------------------------------------------------------------


def _hourly(start, end):
    index = pd.date_range(start, end, freq=pd.offsets.Hour(), tz="UTC")
    return pd.Series(np.arange(len(index), dtype=float), index=index, name="T")


def test_make_blocks_are_chronological_disjoint_and_cover_the_period():
    series = _hourly("2009-12-31 00:00", "2024-01-02 23:00")
    blocks = data.make_blocks(series)

    names = list(blocks)
    assert names == ["train", "val", "calibration", "test"]
    for earlier, later in itertools.pairwise(names):
        assert blocks[earlier].index.max() < blocks[later].index.min()  # no leakage
    assert len(blocks["test"]) == 365 * 24  # 2023 has 365 days
    assert blocks["train"].index.min() == pd.Timestamp("2010-01-01", tz="UTC")
    assert sum(len(b) for b in blocks.values()) == len(
        series.loc["2010-01-01":"2023-12-31 23:00"]
    )


def test_make_blocks_accepts_a_table():
    series = _hourly("2010-01-01 00:00", "2024-01-01 00:00")
    table = pd.DataFrame({"T": series, "U": series * 2})
    blocks = data.make_blocks(table)
    assert isinstance(blocks["test"], pd.DataFrame)
    assert list(blocks["test"].columns) == ["T", "U"]
    assert len(blocks["test"]) == 365 * 24


def test_make_blocks_rejects_bad_boundaries_and_short_series():
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


# --- File table and download --------------------------------------------------------------------


URLS = {
    "H_31_2010-2019.csv.gz": "https://example.org/b",
    "H_31_previous-2020-2024.csv.gz": "https://example.org/c",
    "H_31_2000-2009.csv.gz": "https://example.org/a",
    "H_31_latest-2025-2026.csv.gz": "https://example.org/d",
}


def test_files_from_urls_sorts_by_period_and_needs_a_period():
    files = data.files_from_urls(URLS)
    assert files["first_year"].tolist() == [2000, 2010, 2020, 2025]
    assert files["last_year"].tolist() == [2009, 2019, 2024, 2026]
    with pytest.raises(ValueError, match="No period"):
        data.files_from_urls({"H_31.csv.gz": "https://example.org/x"})


def test_select_files_keeps_the_files_overlapping_the_range():
    files = data.files_from_urls(URLS)
    assert data.select_files(files, "2015", "2021")["first_year"].tolist() == [
        2010,
        2020,
    ]
    assert len(data.select_files(files)) == 4  # no bounds: everything
    assert data.select_files(files, "2024-12-31", None)["first_year"].tolist() == [
        2020,
        2025,
    ]
    with pytest.raises(ValueError, match="No file covers"):
        data.select_files(files, "1900", "1950")


def test_list_files_reads_the_dataset_description(monkeypatch):
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
    body = json.dumps({"resources": {"href": "..."}}).encode()
    monkeypatch.setattr(
        data.urllib.request, "urlopen", lambda request, timeout=None: io.BytesIO(body)
    )
    with pytest.raises(TypeError, match="files_from_urls"):
        data.list_files("31")


def test_download_file_saves_then_uses_the_cache(tmp_path, monkeypatch):
    payload = gzip.compress(b"a;b\n1;2\n")
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        return io.BytesIO(payload)

    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "raw" / "f.csv.gz"

    assert data.download_file("https://example.org/f", dest) == dest
    assert dest.read_bytes() == payload
    data.download_file("https://example.org/f", dest)
    assert len(calls) == 1  # second call did not download again


def test_download_file_refuses_html_pages_and_non_https_links(tmp_path, monkeypatch):
    monkeypatch.setattr(
        data.urllib.request,
        "urlopen",
        lambda request, timeout=None: io.BytesIO(b"<html>oops"),
    )
    dest = tmp_path / "f.csv.gz"

    with pytest.raises(RuntimeError, match="gzip"):
        data.download_file("https://example.org/f", dest)
    assert not dest.exists()
    assert not dest.with_name("f.csv.gz.part").exists()  # no leftover temporary file

    with pytest.raises(ValueError, match="https"):
        data.download_file("http://example.org/f", dest)
    with pytest.raises(ValueError, match="https"):
        data.download_file("file:///etc/passwd", dest)


def test_download_files_uses_safe_names(tmp_path, monkeypatch):
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
    assert (
        data.download_files(nested, tmp_path)[0].parent == tmp_path
    )  # "/" neutralised
    evil = data.files_from_urls({"../x_2010-2019.csv.gz": "https://example.org/a"})
    with pytest.raises(ValueError, match="Unexpected file name"):
        data.download_files(evil, tmp_path)
    bad = data.files_from_urls({".hidden_2010-2019.txt": "https://example.org/a"})
    with pytest.raises(ValueError, match="Unexpected file name"):
        data.download_files(bad, tmp_path)


def test_extract_end_to_end_with_a_fake_server(tmp_path, monkeypatch):
    old = _rows(BLAGNAC, "BLAGNAC", "2009-12-31 22:00", [1.0, 2.0, 3.0, 4.0])
    new = _rows(
        BLAGNAC, "BLAGNAC", "2010-01-01 00:00", [3.5, 4.5, 5.5]
    )  # overlaps `old`
    payloads = {
        "https://example.org/old": _gz_bytes(old),
        "https://example.org/new": _gz_bytes(new),
    }
    asked = []

    def fake_urlopen(request, timeout=None):
        asked.append(request.full_url)
        return io.BytesIO(payloads[request.full_url])

    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    files = data.files_from_urls(
        {
            "H_31_2000-2009.csv.gz": "https://example.org/old",
            "H_31_2010-2019.csv.gz": "https://example.org/new",
        }
    )

    frame = data.extract(BLAGNAC, ["T"], start="2009", files=files, raw_dir=tmp_path)

    assert asked == ["https://example.org/old", "https://example.org/new"]
    assert frame["T"].tolist() == [1.0, 2.0, 3.5, 4.5, 5.5]  # overlap: later file wins
