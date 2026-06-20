from pathlib import Path

import pytest

from xpm_mlboard import (
    MonitoringService,
    SymlinkMonitoringService,
    TensorboardService,
)
from experimaestro.scheduler.services import ProcessWebService, Service


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
    service = TensorboardService(tmp_path / "runs")
    full = service.full_state_dict()
    rebuilt = Service.from_state_dict(full["class"], full["state_dict"])
    assert isinstance(rebuilt, TensorboardService)
    assert rebuilt.path == service.path


def test_build_command(tmp_path: Path):
    service = TensorboardService(tmp_path / "runs")
    cmd = service._build_command()
    assert "tensorboard.main" in cmd
    assert str((tmp_path / "runs").absolute()) in cmd
    assert "--logdir" in cmd


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
