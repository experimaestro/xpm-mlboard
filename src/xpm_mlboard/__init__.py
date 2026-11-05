"""xpm-mlboard: lightweight experimaestro services to monitor ML learning curves.

Exposes a backend-agnostic :class:`MonitoringService` abstraction, a
TensorBoard implementation and a table viewer (:class:`TableViewerService`). These services are meant to be wired into an
experiment by a helper provided by the consuming project (e.g. xpm-torch).
"""

from .backends.tensorboard import TensorboardService as TensorboardService
from .backends.wandb import WandbSyncService as WandbSyncService
from .tables import TableViewerService as TableViewerService
from .service import (
    MonitoringService as MonitoringService,
    SymlinkMonitoringService as SymlinkMonitoringService,
)
