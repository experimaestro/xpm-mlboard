"""xpm-mlboard: lightweight experimaestro services to monitor ML learning curves.

Exposes a backend-agnostic :class:`MonitoringService` abstraction and a
TensorBoard implementation. These services are meant to be wired into an
experiment by a helper provided by the consuming project (e.g. xpm-torch).
"""

from .backends.tensorboard import TensorboardService as TensorboardService
from .service import (
    MonitoringService as MonitoringService,
    SymlinkMonitoringService as SymlinkMonitoringService,
)
