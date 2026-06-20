"""Core monitoring-service abstraction for xpm-mlboard.

A *monitoring service* aggregates the per-task run directories produced during
an experimaestro experiment and exposes them through a visualization backend
(TensorBoard, Weights & Biases, ...).

Concrete backends combine :class:`MonitoringService` with an experimaestro
service implementation that knows how to expose a URL, such as
:class:`experimaestro.scheduler.services.ProcessWebService`.
"""

import abc
import logging
from pathlib import Path

from experimaestro import RunMode, Task, tagspath
from experimaestro.scheduler.services import Service
from experimaestro.utils import cleanupdir

logger = logging.getLogger("xpm.mlboard")


class MonitoringService(Service, abc.ABC):
    """Base class for ML monitoring/visualization services.

    The service collects run directories under a single :attr:`path` and exposes
    them through a backend. Subclasses are typically combined with an
    experimaestro web-service base class (e.g. ``ProcessWebService``).
    """

    #: Unique service identifier (set by concrete backends)
    id: str

    def __init__(self, path: Path):
        # Cooperative init: forwards to the experimaestro Service / web-service
        # base classes (all of which take no constructor arguments).
        super().__init__()

        #: Root directory aggregating every run directory
        self.path = path
        #: Whether the service is actually collecting data
        self.active = False
        logger.info("Monitoring data path is %s", self.path)

    def set_experiment(self, xp):
        super().set_experiment(xp)
        # Only collect data and clean up when running normally
        if xp.run_mode == RunMode.NORMAL:
            self.active = True
            cleanupdir(self.path)
            self.path.mkdir(exist_ok=True, parents=True)
            self.on_active()

    def on_active(self):
        """Hook called once when the service becomes active (NORMAL run mode)."""

    def state_dict(self):
        return {"path": self.path}

    @abc.abstractmethod
    def add(self, task: Task, path: Path):
        """Register the run directory ``path`` produced by ``task``."""
        ...


class SymlinkMonitoringService(MonitoringService, abc.ABC):
    """Monitoring service that aggregates runs by symlinking each task's
    (tag-based) run directory into :attr:`path`.

    Suitable for any filesystem-based backend (TensorBoard, local Aim, ...).
    """

    def add(self, task: Task, path: Path):
        if not self.active:
            return
        tag_path = tagspath(task)
        if not tag_path:
            logger.error(
                "The task is not associated with tags: "
                "cannot link to monitoring data"
            )
            return
        source = self.path / tag_path
        if not path.exists():
            logger.info("Creating monitoring target directory %s", path)
            path.mkdir(parents=True, exist_ok=True)
        if not source.is_symlink():
            try:
                source.symlink_to(path)
                logger.info("Symlinked %s to %s", source, path)
            except Exception:
                logger.exception("Cannot symlink %s to %s", source, path)
