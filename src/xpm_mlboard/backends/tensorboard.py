"""TensorBoard monitoring backend.

Runs ``tensorboard`` as an isolated subprocess (letting the OS pick a free port)
and aggregates run directories through symlinks. The ``tensorboard`` package is
only required at runtime (it is launched via ``python -m tensorboard.main``), so
importing this module stays lightweight.
"""

import logging
import re
import time
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

    def _build_command(self) -> list[str]:
        # --port 0 lets TensorBoard bind to any free port; the chosen URL is
        # then read back from its output in _wait_for_ready.
        return [
            executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(self.path.absolute()),
            "--host",
            "localhost",
            "--port",
            "0",
        ]

    def _wait_for_ready(self) -> str:
        """Poll stdout and stderr for TensorBoard's URL announcement."""
        url_pattern = re.compile(r"https?://localhost:\d+\S*")

        # Safety check: if we don't have log files, we can't wait for ready
        if not self.stdout and not self.stderr:
            logging.error("No log files available for TensorBoard - cannot detect URL")
            raise RuntimeError(
                "TensorBoard log files are missing (log_directory is None)"
            )

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
