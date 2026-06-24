"""Weights & Biases monitoring backend.

Exposes :class:`WandbSyncService`, a service that exports a monitoring aggregate
(TensorBoard event files + ``<tagspath>.tags.json`` sidecars) to a Weights &
Biases project by running the driver-side exporter
(:mod:`xpm_mlboard.exporters.wandb`).

``wandb`` and ``tensorboard`` are imported lazily (only when the service runs),
so importing this module stays lightweight.
"""

import logging
import threading
from pathlib import Path
from typing import Optional

from experimaestro.scheduler.services import ServiceState, WebService

logger = logging.getLogger("xpm.mlboard.wandb")


class WandbSyncService(WebService):
    """Exports the monitoring aggregate at ``path`` to a Weights & Biases project.

    Starting the service runs a one-shot export (reading the run directories and
    tag sidecars under ``path``) and then stops itself. Online vs offline is left
    to wandb's own configuration (``WANDB_MODE``); see the package README for
    credentials and self-hosted-server setup.
    """

    id = "wandb"

    def __init__(
        self,
        path: Path,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        mode: Optional[str] = None,
        reconcile: bool = True,
    ):
        super().__init__()
        self.path = path
        self.project = project
        self.entity = entity
        self.mode = mode
        self.reconcile = reconcile

    def description(self):
        return "Weights & Biases sync"

    def _serve(self, running: threading.Event):
        from xpm_mlboard.exporters.wandb import sync

        # No URL to expose: signal readiness immediately, then export.
        running.set()
        try:
            exported = sync(
                self.path,
                project=self.project,
                entity=self.entity,
                mode=self.mode,
                reconcile=self.reconcile,
            )
            logger.info("Exported %d run(s) to Weights & Biases", len(exported))
        except Exception:
            logger.exception("Weights & Biases export failed")
        finally:
            # One-shot: the export is done, so the service stops itself.
            self.state = ServiceState.STOPPED
