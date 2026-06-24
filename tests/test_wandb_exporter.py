"""Tests for the Weights & Biases exporter.

``wandb`` is not a test dependency, so a fake module is injected. TensorBoard
event files are written with tensorboard's own writer (no torch/TF needed).
"""

import json
from pathlib import Path

from xpm_mlboard.exporters import wandb as wexport
from xpm_mlboard.service import TAGS_SUFFIX


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeRun:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.id = kwargs.get("id")
        self.logged = []  # (metrics, step)
        self.summary = {}
        self.finished = False

    def log(self, metrics, step=None):
        self.logged.append((dict(metrics), step))

    def finish(self):
        self.finished = True


class _FakeApiRun:
    def __init__(self, run_id):
        self.id = run_id
        self.deleted = False

    def delete(self):
        self.deleted = True


class _FakeApi:
    def __init__(self, runs):
        self._runs = runs

    def runs(self, path):
        self.path = path
        return self._runs


class _FakeWandb:
    def __init__(self, api_runs=None):
        self.runs = []
        self._api = _FakeApi(api_runs) if api_runs is not None else None

    def init(self, **kwargs):
        run = _FakeRun(**kwargs)
        self.runs.append(run)
        return run

    def Api(self):
        return self._api


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_events(logdir: Path, series):
    """Write a real TensorBoard event file with the given scalar series."""
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    logdir.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(logdir))
    for tag, points in series.items():
        for step, value in points:
            writer.add_event(
                Event(
                    step=step,
                    wall_time=float(step),
                    summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]),
                )
            )
    writer.flush()
    writer.close()


def _write_run(runs_dir: Path, tagspath: str, tags: dict, series):
    """Create an aggregate entry: a run dir with events + its tag sidecar."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    _write_events(runs_dir / tagspath, series)
    sidecar = runs_dir / f"{tagspath}{TAGS_SUFFIX}"
    sidecar.write_text(json.dumps({"tagspath": tagspath, "tags": tags}))


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_run_id_is_deterministic_and_short():
    a = wexport.run_id_for("model=bert_lr=0.001")
    b = wexport.run_id_for("model=bert_lr=0.001")
    c = wexport.run_id_for("model=t5_lr=0.001")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_read_scalars_roundtrip(tmp_path: Path):
    _write_events(tmp_path / "run", {"loss": [(0, 2.0), (1, 1.5)], "acc": [(1, 0.5)]})
    series = wexport.read_scalars(tmp_path / "run")
    assert series["loss"] == [(0, 2.0), (1, 1.5)]
    assert series["acc"] == [(1, 0.5)]


def test_export_run_logs_by_step_and_summary():
    wb = _FakeWandb()
    wexport.export_run(
        wb,
        project="exp",
        name="model=bert",
        run_id="rid1",
        tags=["model=bert"],
        config={"model": "bert"},
        scalars={"loss": [(0, 2.0), (1, 1.0)], "acc": [(1, 0.9)]},
        mode="offline",
    )
    assert len(wb.runs) == 1
    run = wb.runs[0]
    # init got the deterministic id, resume, config/tags and mode
    assert run.kwargs["id"] == "rid1"
    assert run.kwargs["resume"] == "allow"
    assert run.kwargs["mode"] == "offline"
    assert run.kwargs["config"] == {"model": "bert"}
    # one log call per step, tags sharing a step grouped together
    assert run.logged == [({"loss": 2.0}, 0), ({"loss": 1.0, "acc": 0.9}, 1)]
    # final values in the summary; run finished
    assert run.summary == {"loss": 1.0, "acc": 0.9}
    assert run.finished is True


def test_export_run_omits_mode_and_entity_when_unset():
    wb = _FakeWandb()
    wexport.export_run(
        wb,
        project="exp",
        name="r",
        run_id="rid",
        tags=[],
        config={},
        scalars={},
    )
    kwargs = wb.runs[0].kwargs
    assert "mode" not in kwargs
    assert "entity" not in kwargs


def test_sync_exports_every_run(tmp_path: Path):
    runs = tmp_path / "runs"
    _write_run(runs, "model=bert", {"model": "bert"}, {"loss": [(0, 1.0)]})
    _write_run(runs, "model=t5", {"model": "t5"}, {"loss": [(0, 2.0)]})

    wb = _FakeWandb()
    exported = wexport.sync(runs, project="my-exp", wb=wb)

    assert set(exported) == {"model=bert", "model=t5"}
    assert exported["model=bert"] == wexport.run_id_for("model=bert")
    # one W&B run per task, all in the experiment's project, tags carried over
    assert len(wb.runs) == 2
    by_name = {r.kwargs["name"]: r for r in wb.runs}
    assert by_name["model=bert"].kwargs["project"] == "my-exp"
    assert by_name["model=bert"].kwargs["config"] == {"model": "bert"}
    assert by_name["model=bert"].kwargs["tags"] == ["model=bert"]


def test_sync_skips_run_without_event_dir(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    # Sidecar present but no run directory -> skipped, not crashed
    (runs / f"ghost{TAGS_SUFFIX}").write_text(
        json.dumps({"tagspath": "ghost", "tags": {"a": 1}})
    )
    wb = _FakeWandb()
    exported = wexport.sync(runs, project="exp", wb=wb)
    assert exported == {}
    assert wb.runs == []


def test_reconcile_deletes_stale_runs():
    kept = _FakeApiRun("keep")
    stale = _FakeApiRun("stale")
    wb = _FakeWandb(api_runs=[kept, stale])

    wexport.reconcile_project(wb, project="exp", entity=None, keep_ids={"keep"})

    assert kept.deleted is False
    assert stale.deleted is True
