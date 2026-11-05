"""Load the registered table files into Polars data frames.

:class:`TableRegistry` watches the entry files written by
:class:`~xpm_mlboard.tables.TableViewerService` and the table files they point
to. A file is re-read only when it changes (modification time or size), and the
files of a logical table are concatenated with the run and its tags as extra
columns.
"""

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl

from . import ENTRIES_DIR

logger = logging.getLogger("xpm.mlboard.tables")

#: File extension -> format name
EXTENSIONS = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
    ".tsv": "tsv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".json": "json",
    ".arrow": "ipc",
    ".feather": "ipc",
    ".ipc": "ipc",
}

#: Name of the column identifying the run (task tags path) of each row
RUN_COLUMN = "run"


def detect_format(path: Path, format: Optional[str] = None) -> str:
    if format:
        return format
    try:
        return EXTENSIONS[path.suffix.lower()]
    except KeyError:
        raise ValueError(
            f"Cannot detect the table format of {path} (use format=...)"
        ) from None


def read_table(path: Path, format: Optional[str] = None) -> pl.DataFrame:
    """Read a table file into a Polars data frame."""
    fmt = detect_format(path, format)
    if fmt == "parquet":
        return pl.read_parquet(path)
    if fmt == "csv":
        return pl.read_csv(path, infer_schema_length=10_000)
    if fmt == "tsv":
        return pl.read_csv(path, separator="\t", infer_schema_length=10_000)
    if fmt == "jsonl":
        return pl.read_ndjson(path, infer_schema_length=10_000)
    if fmt == "json":
        # A list of records, or a {column: values} mapping
        with path.open("rt") as fp:
            data = json.load(fp)
        return pl.DataFrame(data, infer_schema_length=10_000)
    if fmt == "ipc":
        return pl.read_ipc(path)
    raise ValueError(f"Unknown table format {fmt!r} for {path}")


def _json_value(value: Any) -> Any:
    """Tag values that are not scalars are shown as JSON strings."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def with_run_columns(
    df: pl.DataFrame, run: Optional[str], tags: Dict[str, Any]
) -> pl.DataFrame:
    """Prepend the run and tag columns to ``df``.

    A tag whose name is already a column of the table is renamed ``tag:<name>``
    (the file's own data wins).
    """
    extra: Dict[str, Any] = {}
    if RUN_COLUMN not in df.columns:
        extra[RUN_COLUMN] = run
    for key, value in tags.items():
        column = key if key not in df.columns and key != RUN_COLUMN else f"tag:{key}"
        extra[column] = _json_value(value)

    columns = [
        pl.lit(value, dtype=pl.String if value is None else None).alias(name)
        for name, value in extra.items()
    ]
    return df.select(*columns, pl.all())


def concat_frames(frames: List[pl.DataFrame]) -> pl.DataFrame:
    """Concatenate frames whose columns may differ (missing values are null,
    and differing types are cast to a common one, or to strings)."""
    if not frames:
        return pl.DataFrame()
    try:
        return pl.concat(frames, how="diagonal_relaxed")
    except Exception:
        logger.warning("Incompatible column types: showing them as strings")
        frames = [
            df.with_columns(pl.all().cast(pl.String, strict=False)) for df in frames
        ]
        return pl.concat(frames, how="diagonal_relaxed")


def to_perspective(df: pl.DataFrame) -> pl.DataFrame:
    """Cast the column types Perspective cannot load (nested, binary, duration,
    time, ...) to supported ones."""
    casts = []
    for name, dtype in df.schema.items():
        if isinstance(dtype, (pl.List, pl.Array, pl.Struct, pl.Object)):
            casts.append(
                pl.col(name)
                .map_elements(
                    lambda v: json.dumps(
                        v.to_list() if isinstance(v, pl.Series) else v, default=str
                    ),
                    return_dtype=pl.String,
                )
                .alias(name)
            )
        elif isinstance(dtype, pl.Duration):
            casts.append(pl.col(name).dt.total_milliseconds().cast(pl.Float64))
        elif isinstance(dtype, (pl.Time, pl.Binary, pl.Categorical, pl.Enum)):
            casts.append(pl.col(name).cast(pl.String))
        elif isinstance(dtype, pl.Decimal):
            casts.append(pl.col(name).cast(pl.Float64))
    return df.with_columns(casts) if casts else df


def to_arrow(df: pl.DataFrame) -> bytes:
    """Serialize as Arrow IPC that Perspective can read (no string views)."""
    buffer = io.BytesIO()
    to_perspective(df).write_ipc(buffer, compat_level=pl.CompatLevel.oldest())
    return buffer.getvalue()


@dataclass
class TableSource:
    """One registered file of a logical table."""

    name: str
    path: Path
    format: Optional[str]
    run: Optional[str]
    tags: Dict[str, Any]
    #: (mtime_ns, size) of the file when last loaded
    signature: Optional[Tuple[int, int]] = None
    frame: Optional[pl.DataFrame] = None
    error: Optional[str] = None

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        return "loaded" if self.frame is not None else "pending"

    def refresh(self) -> bool:
        """Reload the file if it changed; returns whether the rows changed."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            changed = self.frame is not None
            self.signature, self.frame, self.error = None, None, None
            return changed

        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self.signature:
            return False
        self.signature = signature
        try:
            frame = read_table(self.path, self.format)
        except Exception as e:
            # Possibly a partially written file: retried at the next change
            logger.warning("Could not read table %s: %s", self.path, e)
            had_rows = self.frame is not None
            self.frame, self.error = None, str(e)
            return had_rows
        self.frame = with_run_columns(frame, self.run, self.tags)
        self.error = None
        return True


@dataclass
class TableRegistry:
    """The logical tables registered under a table viewer service path."""

    root: Path
    #: entry file name -> source
    sources: Dict[str, TableSource] = field(default_factory=dict)

    def refresh(self) -> Set[str]:
        """Scan the entries and files; returns the names of changed tables."""
        changed: Set[str] = set()
        entries_dir = self.root / ENTRIES_DIR
        seen = set()
        for entry_path in sorted(entries_dir.glob("*.json")):
            key = entry_path.name
            seen.add(key)
            if key not in self.sources:
                try:
                    entry = json.loads(entry_path.read_text())
                    self.sources[key] = TableSource(
                        name=entry["name"],
                        path=Path(entry["path"]),
                        format=entry.get("format"),
                        run=entry.get("run"),
                        tags=entry.get("tags") or {},
                    )
                except Exception:
                    logger.exception("Invalid table entry %s", entry_path)
                    continue
                changed.add(self.sources[key].name)

        for key in list(self.sources):
            if key not in seen:
                changed.add(self.sources.pop(key).name)

        for source in self.sources.values():
            if source.refresh():
                changed.add(source.name)
        return changed

    def names(self) -> List[str]:
        return sorted({source.name for source in self.sources.values()})

    def frame(self, name: str) -> pl.DataFrame:
        """The concatenated rows of the logical table ``name``."""
        return concat_frames(
            [
                source.frame
                for source in self.sources.values()
                if source.name == name and source.frame is not None
            ]
        )

    def summary(self) -> List[Dict[str, Any]]:
        """Per table, the status of each registered file."""
        tables: Dict[str, List[Dict[str, Any]]] = {}
        for source in self.sources.values():
            tables.setdefault(source.name, []).append(
                {
                    "run": source.run,
                    "path": str(source.path),
                    "status": source.status,
                    "rows": 0 if source.frame is None else source.frame.height,
                    "error": source.error,
                }
            )
        return [
            {"name": name, "files": sorted(files, key=lambda f: f["run"] or "")}
            for name, files in sorted(tables.items())
        ]
