"""Service paths: relative to the experiment results path, inactive in dry-run"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from experimaestro import RunMode

from xpm_mlboard import TableViewerService, TensorboardService


class _XP:
    def __init__(self, tmp_path: Path, run_mode: RunMode):
        self.name = "xp"
        self.run_mode = run_mode
        self.workspace = SimpleNamespace(scheduler_services_path=tmp_path / "svc")
        self._results = tmp_path / "results"

    @property
    def resultspath(self):
        if self.run_mode != RunMode.NORMAL:
            raise RuntimeError("xp.resultspath is not available")
        return self._results


@pytest.mark.parametrize(
    "cls, path, expected",
    [
        (TableViewerService, None, "results/tables"),
        (TensorboardService, None, "results/runs"),
        (TableViewerService, "diag/tables", "results/diag/tables"),
    ],
)
def test_relative_path(tmp_path, cls, path, expected):
    service = cls(path)
    service.set_experiment(_XP(tmp_path, RunMode.NORMAL))
    assert service.active and service.path == tmp_path / expected
    assert service.path.is_dir()


def test_absolute_path(tmp_path):
    service = TableViewerService(tmp_path / "elsewhere")
    service.set_experiment(_XP(tmp_path, RunMode.NORMAL))
    assert service.path == tmp_path / "elsewhere"


def test_dry_run(tmp_path):
    """No access to the results path, nothing created, registrations ignored"""
    service = TableViewerService()
    service.set_experiment(_XP(tmp_path, RunMode.DRY_RUN))
    assert not service.active
    assert not (tmp_path / "results").exists()
    service.add_table("summary", tmp_path / "summary.csv")
    assert not (tmp_path / "results").exists()
