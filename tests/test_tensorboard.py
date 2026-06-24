from pathlib import Path

import pytest

from xpm_mlboard import (
    MonitoringService,
    SymlinkMonitoringService,
    TensorboardService,
)
from xpm_mlboard.service import TAGS_SUFFIX, read_run_tags
from experimaestro import Config, Param, tag, tagspath
from experimaestro.scheduler.services import ProcessWebService, Service


class _DummyConfig(Config):
    lr: Param[float]
    model: Param[str]


def test_mro_and_identity():
    """The backend composes the monitoring + process-web-service bases."""
    assert issubclass(TensorboardService, SymlinkMonitoringService)
    assert issubclass(TensorboardService, MonitoringService)
    assert issubclass(TensorboardService, ProcessWebService)
    assert TensorboardService.id == "tensorboard"


def test_instantiation(tmp_path: Path):
    service = TensorboardService(tmp_path / "runs")
    assert service.path == tmp_path / "runs"
    assert service.active is False
    assert service.description() == "TensorBoard service"


def test_state_dict_roundtrip(tmp_path: Path):
    """state_dict feeds Service.from_state_dict to recreate the instance."""
    service = TensorboardService(tmp_path / "runs", log_directory=tmp_path / "logs")
    state = service.state_dict()
    # Base service state (log_directory) is preserved alongside our path
    assert state["path"] == tmp_path / "runs"
    assert state["log_directory"] == tmp_path / "logs"

    full = service.full_state_dict()
    rebuilt = Service.from_state_dict(full["class"], full["state_dict"])
    assert isinstance(rebuilt, TensorboardService)
    assert rebuilt.path == service.path
    assert rebuilt.log_directory == service.log_directory


def test_build_command(tmp_path: Path):
    service = TensorboardService(tmp_path / "runs")
    cmd = service._build_command()
    assert "tensorboard.main" in cmd
    assert str((tmp_path / "runs").absolute()) in cmd
    assert "--logdir" in cmd
    # OS-assigned port
    assert cmd[cmd.index("--port") + 1] == "0"


def test_add_inactive_is_noop(tmp_path: Path):
    """When not active (e.g. dry runs), add() must not touch the filesystem."""
    service = TensorboardService(tmp_path / "runs")
    assert service.active is False
    # Should return without raising even with a dummy task
    service.add(task=None, path=tmp_path / "task")
    assert not (tmp_path / "task").exists()


def test_monitoring_service_is_abstract():
    with pytest.raises(TypeError):
        MonitoringService(Path("/tmp/x"))  # abstract add()


def test_add_registers_tags(tmp_path: Path):
    """add() symlinks the run and writes a backend-agnostic tag sidecar."""
    service = TensorboardService(tmp_path / "runs")
    service.path.mkdir(parents=True)
    service.active = True  # normally set by set_experiment in NORMAL run mode

    cfg = _DummyConfig.C(lr=tag(0.001), model=tag("bert"))
    tp = tagspath(cfg)

    service.add(task=cfg, path=tmp_path / "job" / "runs")

    # The run is symlinked and the tag sidecar is written next to it
    assert (service.path / tp).is_symlink()
    assert (service.path / f"{tp}{TAGS_SUFFIX}").exists()

    # read_run_tags recovers exactly this experiment's tags for the run
    assert read_run_tags(service.path) == {tp: {"lr": 0.001, "model": "bert"}}


def test_read_run_tags_missing_dir(tmp_path: Path):
    assert read_run_tags(tmp_path / "does-not-exist") == {}
