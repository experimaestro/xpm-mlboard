"""TensorBoard monitoring backend.

Runs ``tensorboard`` as an isolated subprocess on a free port and aggregates
run directories through symlinks. The ``tensorboard`` package is only required
at runtime (it is launched via ``python -m tensorboard.main``), so importing
this module stays lightweight.
"""

import logging
import re
import socket
import time
from pathlib import Path
from sys import executable

from experimaestro.scheduler.services import ProcessWebService

from ..service import SymlinkMonitoringService

logger = logging.getLogger("xpm.mlboard.tensorboard")


class TensorboardService(SymlinkMonitoringService, ProcessWebService):
    """Monitoring backend backed by a TensorBoard subprocess."""

    id = "tensorboard"

    def on_active(self):
        logger.info("You can monitor learning with:")
        logger.info("tensorboard --logdir=%s", self.path)

    def description(self):
        return "TensorBoard service"

    @staticmethod
    def _find_available_port(start: int = 6006, max_tries: int = 100) -> int:
        for port in range(start, start + max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", port)) != 0:
                    return port
        return 0

    def _build_command(self) -> list[str]:
        port = self._find_available_port()
        return [
            executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(self.path.absolute()),
            "--host",
            "localhost",
            "--port",
            str(port),
        ]

    def _wait_for_ready(self) -> str:
        """Poll stdout and stderr for TensorBoard's URL announcement."""
        url_pattern = re.compile(r"https?://localhost:\d+\S*")
        while True:
            if self.process and self.process.poll() is not None:
                # Read any error output for diagnostics
                err = ""
                if self.stderr and self.stderr.exists():
                    err = self.stderr.read_text()
                raise RuntimeError(
                    f"TensorBoard exited with code {self.process.returncode}: {err}"
                )
            # Check both stdout and stderr for the URL
            for log_path in (self.stderr, self.stdout):
                if log_path and log_path.exists():
                    content = log_path.read_text()
                    if match := url_pattern.search(content):
                        return match.group(0)
            time.sleep(0.2)
