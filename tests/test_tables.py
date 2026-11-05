"""Tests for the table viewer service, its loader and its web server."""

import json
import os
from pathlib import Path

import pytest
from experimaestro import Config, Param, tag, tagspath
from experimaestro.scheduler.services import ProcessWebService, Service

from xpm_mlboard import MonitoringService, TableViewerService
from xpm_mlboard.tables import ENTRIES_DIR, VIEWS_DIR

pl = pytest.importorskip("polars")


class _Eval(Config):
    lr: Param[float]
    model: Param[str]


def _active_service(tmp_path: Path) -> TableViewerService:
    service = TableViewerService(tmp_path / "tables")
    service.path.mkdir(parents=True)
    service.active = True  # normally set by set_experiment in NORMAL run mode
    return service


def _entries(service: TableViewerService) -> list:
    return [
        json.loads(p.read_text())
        for p in sorted((service.path / ENTRIES_DIR).glob("*.json"))
    ]


def _touch_later(path: Path):
    """Make sure a rewritten file has a new modification time."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


# --- Service -------------------------------------------------------------


def test_identity_and_state(tmp_path: Path):
    assert issubclass(TableViewerService, MonitoringService)
    assert issubclass(TableViewerService, ProcessWebService)
    assert TableViewerService.id == "tables"

    service = TableViewerService(
        tmp_path / "tables", log_directory=tmp_path / "logs", project="exp"
    )
    full = service.full_state_dict()
    rebuilt = Service.from_state_dict(full["class"], full["state_dict"])
    assert isinstance(rebuilt, TableViewerService)
    assert rebuilt.path == service.path
    assert rebuilt.log_directory == service.log_directory


def test_build_command(tmp_path: Path):
    cmd = TableViewerService(tmp_path / "tables")._build_command()
    assert cmd[1:3] == ["-m", "xpm_mlboard.tables"]
    assert cmd[cmd.index("--root") + 1] == str((tmp_path / "tables").absolute())
    assert cmd[cmd.index("--port") + 1] == "0"


def test_add_inactive_is_noop(tmp_path: Path):
    service = TableViewerService(tmp_path / "tables")
    service.add(task=None, path=tmp_path / "eval.csv")
    service.add_table("summary", tmp_path / "summary.csv")
    assert not service.path.exists()


def test_add_registers_entries(tmp_path: Path):
    service = _active_service(tmp_path)
    cfg = _Eval.C(lr=tag(0.01), model=tag("alpha"))
    service.add(cfg, tmp_path / "job" / "eval.parquet", name="eval")
    service.add(cfg, tmp_path / "job" / "metrics.csv")  # name from the stem
    service.add_table("summary", tmp_path / "summary.csv", format="csv")

    entries = {(e["name"], e["run"]): e for e in _entries(service)}
    assert set(entries) == {
        ("eval", tagspath(cfg)),
        ("metrics", tagspath(cfg)),
        ("summary", None),
    }
    assert entries["eval", tagspath(cfg)]["tags"] == {"lr": 0.01, "model": "alpha"}
    assert entries["summary", None]["format"] == "csv"
    assert entries["summary", None]["path"] == str(tmp_path / "summary.csv")


def test_add_same_tags_keeps_first(tmp_path: Path, caplog):
    service = _active_service(tmp_path)
    first, second = tmp_path / "job1" / "eval.csv", tmp_path / "job2" / "eval.csv"
    service.add(_Eval.C(lr=0.1, model=tag("alpha")), first, name="eval")

    with caplog.at_level("WARNING", logger="xpm.mlboard.tables"):
        # Re-adding the same file is silent
        service.add(_Eval.C(lr=0.1, model=tag("alpha")), first, name="eval")
        assert not caplog.records
        service.add(_Eval.C(lr=0.2, model=tag("alpha")), second, name="eval")
        assert len(caplog.records) == 1
        assert "is not shown" in caplog.records[0].getMessage()

    assert [e["path"] for e in _entries(service)] == [str(first)]


def test_cleanup_keeps_views(tmp_path: Path):
    service = _active_service(tmp_path)
    service.add_table("summary", tmp_path / "summary.csv")
    view = service.path / VIEWS_DIR / "summary.json"
    view.parent.mkdir()
    view.write_text("{}")

    service.cleanup()
    assert not _entries(service)
    assert view.exists()


# --- Loader --------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["parquet", "csv", "tsv", "jsonl", "json", "ipc"])
def test_read_formats(tmp_path: Path, fmt: str):
    from xpm_mlboard.tables.loader import read_table

    df = pl.DataFrame({"metric": ["mrr", "ndcg"], "value": [0.5, 0.25]})
    path = tmp_path / f"table.{fmt}"
    if fmt == "parquet":
        df.write_parquet(path)
    elif fmt == "csv":
        df.write_csv(path)
    elif fmt == "tsv":
        df.write_csv(path, separator="\t")
    elif fmt == "jsonl":
        df.write_ndjson(path)
    elif fmt == "json":
        path.write_text(json.dumps(df.to_dicts()))
    else:
        df.write_ipc(path)

    assert read_table(path).equals(df)


def test_unknown_extension(tmp_path: Path):
    from xpm_mlboard.tables.loader import detect_format

    with pytest.raises(ValueError, match="format="):
        detect_format(tmp_path / "table.xyz")
    assert detect_format(tmp_path / "table.xyz", "csv") == "csv"


def test_run_columns_and_collisions():
    from xpm_mlboard.tables.loader import with_run_columns

    df = pl.DataFrame({"model": ["from-data"], "value": [1.0]})
    out = with_run_columns(df, "run-a", {"model": "alpha", "lr": 0.1, "dims": [1, 2]})
    assert out.columns == ["run", "tag:model", "lr", "dims", "model", "value"]
    row = out.row(0, named=True)
    assert row["run"] == "run-a"
    assert row["model"] == "from-data"  # the file's own column wins
    assert row["tag:model"] == "alpha"
    assert row["dims"] == "[1, 2]"


def test_concat_and_perspective_types():
    import datetime

    from xpm_mlboard.tables.loader import concat_frames, to_perspective

    a = pl.DataFrame({"run": ["a"], "value": [1]})
    b = pl.DataFrame({"run": ["b"], "value": [0.5], "extra": ["x"]})
    out = concat_frames([a, b])
    assert out.columns == ["run", "value", "extra"]
    assert out["value"].to_list() == [1.0, 0.5]
    assert out["extra"].to_list() == [None, "x"]

    df = pl.DataFrame(
        {
            "list": [[1, 2]],
            "struct": [{"k": 1}],
            "binary": [b"x"],
            "duration": [datetime.timedelta(seconds=2)],
            "time": [datetime.time(1, 2)],
        }
    )
    converted = to_perspective(df).row(0, named=True)
    assert converted["list"] == "[1, 2]"
    assert json.loads(converted["struct"]) == {"k": 1}
    assert converted["duration"] == 2000.0
    assert isinstance(converted["binary"], str)
    assert isinstance(converted["time"], str)


def test_registry_refresh(tmp_path: Path):
    from xpm_mlboard.tables.loader import TableRegistry

    service = _active_service(tmp_path)
    done, later = tmp_path / "a" / "eval.csv", tmp_path / "b" / "eval.csv"
    done.parent.mkdir()
    done.write_text("metric,value\nmrr,0.5\n")
    service.add(_Eval.C(lr=tag(0.1), model=tag("alpha")), done, name="eval")
    service.add(_Eval.C(lr=tag(0.2), model=tag("beta")), later, name="eval")

    registry = TableRegistry(service.path)
    assert registry.refresh() == {"eval"}
    assert registry.names() == ["eval"]
    frame = registry.frame("eval")
    assert frame.height == 1
    assert frame.row(0, named=True)["model"] == "alpha"
    statuses = sorted(f["status"] for f in registry.summary()[0]["files"])
    assert statuses == ["loaded", "pending"]

    # Nothing changed
    assert registry.refresh() == set()

    # The pending run produces its results (with a new column)
    later.parent.mkdir()
    later.write_text("metric,value,seconds\nmrr,0.25,3\n")
    assert registry.refresh() == {"eval"}
    frame = registry.frame("eval").sort("lr")
    assert frame["value"].to_list() == [0.5, 0.25]
    assert frame["seconds"].to_list() == [None, 3]

    # A rewritten file replaces the rows of its run
    done.write_text("metric,value\nmrr,0.75\nndcg,0.5\n")
    _touch_later(done)
    assert registry.refresh() == {"eval"}
    assert registry.frame("eval").height == 3

    # Unreadable file: reported as an error, its rows are dropped
    (source,) = [s for s in registry.sources.values() if s.path == done]
    source.format = "parquet"  # the CSV content is not valid Parquet
    _touch_later(done)
    assert registry.refresh() == {"eval"}
    assert "error" in {f["status"] for f in registry.summary()[0]["files"]}
    assert registry.frame("eval").height == 1


# --- Server --------------------------------------------------------------


def test_publisher_generations():
    perspective = pytest.importorskip("perspective")
    from xpm_mlboard.tables.server import Publisher

    publisher = Publisher(perspective.Server().new_local_client())
    publisher.publish("eval", pl.DataFrame({"run": ["a"], "value": [1.0]}))
    first = publisher.tables["eval"]
    assert first.hosted_name == "eval#0"
    assert first.types == {"run": "string", "value": "float"}

    # Same columns: rows replaced in place
    publisher.publish("eval", pl.DataFrame({"run": ["a", "b"], "value": [1.0, 2.0]}))
    assert publisher.tables["eval"] is first
    assert first.table.size() == 2

    # New column: next generation, the previous one is deleted
    publisher.publish(
        "eval", pl.DataFrame({"run": ["a"], "value": [1.0], "seconds": [3]})
    )
    assert publisher.tables["eval"].hosted_name == "eval#1"
    assert len(publisher.retired) == 1
    publisher.collect()
    assert not publisher.retired

    # No rows anymore
    publisher.publish("eval", pl.DataFrame())
    assert "eval" not in publisher.tables


def test_api(tmp_path: Path):
    pytest.importorskip("perspective")
    from fastapi.testclient import TestClient

    from xpm_mlboard.tables.server import TablesApp, create_app

    service = _active_service(tmp_path)
    results = tmp_path / "job" / "eval.jsonl"
    results.parent.mkdir()
    results.write_text('{"metric": "mrr", "value": 0.5}\n')
    service.add(_Eval.C(lr=tag(0.1), model=tag("alpha")), results, name="eval")

    tables = TablesApp(service.path)
    tables.sync()
    with TestClient(create_app(tables)) as client:
        assert "<title>Experiment tables</title>" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/perspective.js").status_code == 200

        (table,) = client.get("/api/tables").json()["tables"]
        assert table["name"] == "eval"
        assert table["table"] == "eval#0"
        assert table["rows"] == 1
        assert table["types"]["lr"] == "float"

        # Views are stored per logical table (names may contain slashes)
        assert client.get("/api/views/a/b").json() is None
        view = {"plugin": "Datagrid", "sort": [["value", "desc"]]}
        assert client.put("/api/views/a/b", json=view).status_code == 200
        assert client.get("/api/views/a/b").json() == view
        assert (service.path / VIEWS_DIR / "a%2Fb.json").exists()
