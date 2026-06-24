"""Weights & Biases exporter.

Reads a monitoring aggregate directory (the symlinked TensorBoard run
directories plus the ``<tagspath>.tags.json`` sidecars written by
:class:`xpm_mlboard.SymlinkMonitoringService`) and creates/updates one W&B run
per task, attaching that experiment's tags as the run's ``config``/``tags``.

The exporter runs **driver-side**: tasks never talk to W&B (they only write
event files, which works on offline compute nodes). This module imports neither
``wandb`` nor ``tensorboard`` at import time — both are pulled in lazily so the
rest of xpm-mlboard stays lightweight.

Design notes:

- One W&B run per task, named after its ``tagspath``, with a **deterministic
  run id** derived from the ``tagspath`` so re-exporting (or syncing an offline
  run) updates the same run instead of creating duplicates (``resume="allow"``).
- ``mode`` is left to wandb's own configuration (``WANDB_MODE``) unless given,
  so online vs offline is a deployment choice.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from xpm_mlboard.service import read_run_tags

logger = logging.getLogger("xpm.mlboard.wandb")

#: A scalar series: ``tag -> [(step, value), ...]`` (steps ascending).
ScalarSeries = Dict[str, List[Tuple[int, float]]]


def run_id_for(tagspath: str) -> str:
    """Deterministic W&B run id for a run (so re-exports update in place)."""
    # Not security-sensitive: a stable short id derived from the tag path.
    digest = hashlib.sha1(tagspath.encode("utf-8"), usedforsecurity=False)
    return digest.hexdigest()[:16]


def read_scalars(run_dir: Path) -> ScalarSeries:
    """Read every scalar series from the TensorBoard event files in ``run_dir``."""
    from tensorboard.backend.event_processing.event_accumulator import (
        SCALARS,
        EventAccumulator,
    )

    acc = EventAccumulator(str(run_dir), size_guidance={SCALARS: 0})
    acc.Reload()
    series: ScalarSeries = {}
    for tag in acc.Tags().get("scalars", []):
        series[tag] = [(int(s.step), float(s.value)) for s in acc.Scalars(tag)]
    return series


def _resolve_wandb(wb):
    if wb is not None:
        return wb
    try:
        import wandb
    except ImportError as e:  # pragma: no cover - exercised via message only
        raise ImportError(
            "The 'wandb' package is required to export to Weights & Biases. "
            "Install it with: pip install 'xpm-mlboard[wandb]'"
        ) from e
    return wandb


def export_run(
    wb,
    *,
    project: str,
    name: str,
    run_id: str,
    tags: List[str],
    config: Dict[str, object],
    scalars: ScalarSeries,
    mode: Optional[str] = None,
    entity: Optional[str] = None,
) -> str:
    """Create or update a single W&B run from a scalar series.

    :param wb: the ``wandb`` module (injected for testability).
    :return: the run id that was written.
    """
    init_kwargs = dict(
        project=project,
        id=run_id,
        name=name,
        tags=list(tags),
        config=dict(config),
        resume="allow",
        reinit=True,
    )
    if entity is not None:
        init_kwargs["entity"] = entity
    if mode is not None:
        init_kwargs["mode"] = mode

    run = wb.init(**init_kwargs)
    try:
        # Group points by step so all tags sharing a step are logged together
        # and the W&B step stays monotonically increasing.
        by_step: Dict[int, Dict[str, float]] = {}
        for tag, points in scalars.items():
            for step, value in points:
                by_step.setdefault(int(step), {})[tag] = float(value)
        for step in sorted(by_step):
            run.log(by_step[step], step=step)

        # Final value of each metric goes to the run summary.
        for tag, points in scalars.items():
            if points:
                run.summary[tag] = float(points[-1][1])
    finally:
        run.finish()
    return run_id


def reconcile_project(wb, *, project: str, entity: Optional[str], keep_ids) -> None:
    """Delete W&B runs in ``project`` whose id is not in ``keep_ids``.

    Used after a full export so that runs whose task left the experiment do not
    linger. Requires API access (online).
    """
    path = f"{entity}/{project}" if entity else project
    try:
        runs = wb.Api().runs(path)
    except Exception:
        logger.exception("Could not list runs for reconciliation of %s", path)
        return
    keep = set(keep_ids)
    for run in runs:
        if run.id not in keep:
            logger.info("Reconcile: deleting stale W&B run %s", run.id)
            try:
                run.delete()
            except Exception:
                logger.exception("Could not delete W&B run %s", run.id)


def sync(
    runs_dir,
    *,
    project: str,
    mode: Optional[str] = None,
    entity: Optional[str] = None,
    reconcile: bool = False,
    wb=None,
) -> Dict[str, str]:
    """Export every run under ``runs_dir`` to the W&B ``project``.

    :param runs_dir: a monitoring aggregate directory (the service ``path``).
    :param project: the W&B project (one project per experiment).
    :param mode: wandb mode (``online``/``offline``/``disabled``); defaults to
        wandb's own configuration (``WANDB_MODE``).
    :param reconcile: if true, delete project runs whose task is no longer present.
    :param wb: the ``wandb`` module (injected for testability; lazily imported).
    :return: a mapping ``tagspath -> run id`` for every exported run.
    """
    runs_dir = Path(runs_dir)
    wb = _resolve_wandb(wb)

    run_tags = read_run_tags(runs_dir)
    exported: Dict[str, str] = {}
    for tagspath, tags in sorted(run_tags.items()):
        run_dir = runs_dir / tagspath
        if not run_dir.exists():
            logger.warning("No run directory for %s (expected %s)", tagspath, run_dir)
            continue
        run_id = run_id_for(tagspath)
        export_run(
            wb,
            project=project,
            name=tagspath,
            run_id=run_id,
            tags=[f"{key}={value}" for key, value in sorted(tags.items())],
            config=tags,
            scalars=read_scalars(run_dir),
            mode=mode,
            entity=entity,
        )
        exported[tagspath] = run_id
        logger.info("Exported run %s -> %s", tagspath, run_id)

    if reconcile:
        reconcile_project(
            wb, project=project, entity=entity, keep_ids=set(exported.values())
        )
    return exported
