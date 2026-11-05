"""Table viewer service: browse the tables produced by an experiment.

Tasks (or the experiment itself) register tabular files — Parquet, CSV/TSV,
JSON lines, JSON records or Arrow IPC — under a *logical table name*. Every file
registered under the same name is concatenated into one table, with the task's
experimaestro tags added as columns, and served through a
`Perspective <https://perspective-dev.github.io>`_ viewer (pivot, group/split by
tag, charts).

The service only writes small entry files (one per registered file) under
``<path>/entries``; the viewer subprocess (``python -m xpm_mlboard.tables``)
watches them and the registered files, so tables show up and refresh while the
experiment runs. This module stays lightweight: Polars, Perspective and Tornado
are only imported by the viewer process (``tables`` extra).
"""

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from sys import executable
from typing import Any, Dict, Optional

from experimaestro import Task, tags, tagspath
from experimaestro.scheduler.services import ProcessWebService
from experimaestro.utils import cleanupdir

from ..service import MonitoringService

logger = logging.getLogger("xpm.mlboard.tables")

#: Sub-directory of the service path holding the registered table entries
ENTRIES_DIR = "entries"
#: Sub-directory of the service path holding the saved viewer configurations
#: (kept across experiment runs)
VIEWS_DIR = "views"
#: Line printed by the viewer process once it serves requests
READY_MESSAGE = "xpm-mlboard tables viewer at"


def entry_id(name: str, key: str) -> str:
    """Stable file name for the entry of ``key`` (run or path) in table ``name``."""
    return hashlib.sha1(json.dumps([name, key]).encode()).hexdigest()[:20]


def write_entry(root: Path, entry: Dict[str, Any]) -> Path:
    """Atomically write an entry file under ``root/entries``."""
    key = entry["run"] if entry.get("run") is not None else entry["path"]
    target = root / ENTRIES_DIR / f"{entry_id(entry['name'], key)}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wt") as fp:
        json.dump(entry, fp, indent=2, sort_keys=True)
    tmp.replace(target)
    return target


class TableViewerService(MonitoringService, ProcessWebService):
    """Serves the tables registered by the experiment in a Perspective viewer.

    Usage::

        # Under xp.resultspath / "tables" (default; also works in dry-run
        # mode, where the service stays inactive)
        tables = xp.add_service(TableViewerService())
        task = Evaluate(...).submit()
        tables.add(task, task.results_path, name="eval")
        tables.add_table("summary", xp.resultspath / "summary.csv")

    A file is (re)loaded whenever it changes, so a task can rewrite its results
    file: the rows of that run are replaced.
    """

    id = "tables"
    default_path = "tables"

    def description(self):
        return "Table viewer"

    def cleanup(self):
        # Registered tables are per run, but saved views are kept
        cleanupdir(self.path / ENTRIES_DIR)

    def add(
        self,
        task: Task,
        path: Path,
        name: Optional[str] = None,
        format: Optional[str] = None,
    ):
        """Register the table file ``path`` produced by ``task``.

        :param task: The task producing the file; its experimaestro tags are
            added as columns, and its tags path identifies its rows (``run``
            column).
        :param path: The table file; it does not need to exist yet.
        :param name: Logical table name (files with the same name are
            concatenated); defaults to the file stem.
        :param format: File format (``parquet``, ``csv``, ``tsv``, ``jsonl``,
            ``json``, ``ipc``); detected from the extension by default.
        """
        if not self.active:
            return
        run = tagspath(task)
        if not run:
            logger.error(
                "Task is not associated with tags: cannot register table %s", path
            )
            return
        self._register(path, name, format, run=run, tags=dict(tags(task)))

    def add_table(self, name: str, path: Path, format: Optional[str] = None):
        """Register a table file that is not tied to a task (e.g. a summary
        written by the experiment itself)."""
        if not self.active:
            return
        self._register(path, name, format, run=None, tags={})

    def _register(
        self,
        path: Path,
        name: Optional[str],
        format: Optional[str],
        run: Optional[str],
        tags: Dict[str, Any],
    ):
        path = Path(path).absolute()
        name = name or path.stem
        entry = {
            "name": name,
            "path": str(path),
            "format": format,
            "run": run,
            "tags": tags,
        }

        key = run if run is not None else str(path)
        existing = self.path / ENTRIES_DIR / f"{entry_id(name, key)}.json"
        if existing.exists():
            try:
                previous = json.loads(existing.read_text())
            except Exception:
                previous = None
            if previous is not None and previous.get("path") != entry["path"]:
                # Same as SymlinkMonitoringService: rows of runs with the same
                # tags could not be told apart, so only the first is kept.
                logger.warning(
                    "Table %s of run %s is not shown: %s is already registered "
                    "for this run (same tags); tag the tasks with the "
                    "parameters that differ",
                    path,
                    run,
                    previous.get("path"),
                )
                return

        try:
            write_entry(self.path, entry)
        except Exception:
            logger.exception("Could not register table %s (%s)", name, path)

    def _build_command(self) -> list[str]:
        return [
            executable,
            "-m",
            "xpm_mlboard.tables",
            "--root",
            str(self.path.absolute()),
            "--host",
            "localhost",
            "--port",
            "0",
        ]

    def _wait_for_ready(self) -> str:
        pattern = re.compile(re.escape(READY_MESSAGE) + r" (https?://\S+)")
        if not self.stdout and not self.stderr:
            raise RuntimeError("Table viewer log files are missing (no log_directory)")

        while True:
            if self.process and self.process.poll() is not None:
                err = ""
                if self.stderr and self.stderr.exists():
                    err = self.stderr.read_text()
                raise RuntimeError(
                    f"Table viewer exited with code {self.process.returncode}: {err}"
                )
            for log_path in (self.stderr, self.stdout):
                if log_path and log_path.exists():
                    if match := pattern.search(log_path.read_text()):
                        # Proxies append '/<path>' to the service URL
                        return match.group(1).rstrip("/")
            time.sleep(0.2)
