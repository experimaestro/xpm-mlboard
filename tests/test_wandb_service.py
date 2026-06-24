"""Tests for the WandbSyncService and its wiring as a TensorBoard sub-service."""

import threading
from pathlib import Path

from experimaestro.scheduler.services import ServiceState
from xpm_mlboard import TensorboardService, WandbSyncService


def test_wandb_service_basic(tmp_path: Path):
    svc = WandbSyncService(tmp_path, project="exp", entity="me", mode="offline")
    assert svc.id == "wandb"
    assert svc.description() == "Weights & Biases sync"
    assert (svc.project, svc.entity, svc.mode) == ("exp", "me", "offline")


def test_wandb_service_serve_runs_exporter_then_stops(tmp_path: Path, monkeypatch):
    calls = {}

    def fake_sync(path, *, project, entity, mode, reconcile):
        calls["args"] = (Path(path), project, entity, mode, reconcile)
        return {"a": "id1", "b": "id2"}

    import xpm_mlboard.exporters.wandb as wexport

    monkeypatch.setattr(wexport, "sync", fake_sync)

    svc = WandbSyncService(tmp_path, project="exp", mode="offline")
    running = threading.Event()
    svc._serve(running)

    assert running.is_set()
    assert calls["args"] == (tmp_path, "exp", None, "offline", True)
    # one-shot export: the service stops itself
    assert svc.state == ServiceState.STOPPED


def test_wandb_service_serve_stops_even_on_failure(tmp_path: Path, monkeypatch):
    def boom(path, **kwargs):
        raise RuntimeError("nope")

    import xpm_mlboard.exporters.wandb as wexport

    monkeypatch.setattr(wexport, "sync", boom)

    svc = WandbSyncService(tmp_path, project="exp")
    running = threading.Event()
    svc._serve(running)  # must not raise

    assert running.is_set()
    assert svc.state == ServiceState.STOPPED


def test_tensorboard_exposes_wandb_subservice(tmp_path: Path):
    tb = TensorboardService(tmp_path / "runs")
    tb.project = "my-exp"  # normally set by set_experiment

    subs = tb.subservices()
    assert subs[0] is tb  # the service is its own (TensorBoard) sub-service
    assert isinstance(subs[1], WandbSyncService)
    assert subs[1].project == "my-exp"

    # stable instance across calls (lifecycle state preserved)
    assert tb.subservices()[1] is subs[1]
    assert [s.id for s in subs] == ["tensorboard", "wandb"]
