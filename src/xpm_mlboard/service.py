"""Core monitoring-service abstraction for xpm-mlboard.

A *monitoring service* aggregates the per-task run directories produced during
an experimaestro experiment and exposes them through a visualization backend
(TensorBoard, Weights & Biases, ...).

Concrete backends combine :class:`MonitoringService` with an experimaestro
service implementation that knows how to expose a URL, such as
:class:`experimaestro.scheduler.services.ProcessWebService`.
"""

import abc
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from experimaestro import RunMode, Task, tags, tagspath
from experimaestro.scheduler.services import Service
from experimaestro.utils import cleanupdir

logger = logging.getLogger("xpm.mlboard")

#: Suffix of the per-run tag sidecar files written next to each symlinked run
#: directory (e.g. ``model=bert_lr=0.001.tags.json``). Backend-agnostic metadata
#: that downstream consumers (e.g. a Weights & Biases exporter) use to attach
#: each run's experimaestro tags to the corresponding visualization run.
TAGS_SUFFIX = ".tags.json"


def read_run_tags(path: Path) -> Dict[str, dict]:
    """Read the tag sidecars under a monitoring aggregate directory.

    :param path: A monitoring service aggregate directory (its ``self.path``).
    :return: A mapping ``tagspath -> {tag name: value}`` for every registered
        run (empty if the directory does not exist or has no sidecars).
    """
    result: Dict[str, dict] = {}
    if not path.is_dir():
        return result
    for sidecar in path.glob(f"*{TAGS_SUFFIX}"):
        try:
            with sidecar.open("rt") as fp:
                entry = json.load(fp)
            result[entry["tagspath"]] = entry.get("tags", {})
        except Exception:
            logger.exception("Could not read tag sidecar %s", sidecar)
    return result


class MonitoringService(Service, abc.ABC):
    """Base class for ML monitoring/visualization services.

    The service collects run directories under a single :attr:`path` and exposes
    them through a backend. Subclasses are typically combined with an
    experimaestro web-service base class (e.g. ``ProcessWebService``).
    """

    #: Unique service identifier (set by concrete backends)
    id: str

    def __init__(self, path: Path, log_directory: Optional[Path] = None):
        # Cooperative init: forwards log_directory to the experimaestro Service /
        # web-service base classes.
        super().__init__(log_directory=log_directory)

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
        # Preserve the base service state (e.g. log_directory) and add our path.
        state = super().state_dict()
        state["path"] = self.path
        return state

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
                "The task is not associated with tags: cannot link to monitoring data"
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

        # Register the task's tags next to the run so backend-agnostic consumers
        # (e.g. a Weights & Biases exporter) can attach them to the run.
        self._register_tags(task, tag_path)

    def _register_tags(self, task: Task, tag_path: str) -> None:
        """Write ``<tag_path>.tags.json`` recording this task's experimaestro
        tags. One file per run (no shared-file contention); written atomically.
        """
        try:
            entry = {"tagspath": tag_path, "tags": dict(tags(task))}
            sidecar = self.path / f"{tag_path}{TAGS_SUFFIX}"
            tmp = sidecar.with_name(sidecar.name + ".tmp")
            with tmp.open("wt") as fp:
                json.dump(entry, fp, indent=2, sort_keys=True)
            tmp.replace(sidecar)
        except Exception:
            logger.exception("Could not register tags for run %s", tag_path)
