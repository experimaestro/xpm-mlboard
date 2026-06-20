# xpm-mlboard

Lightweight [experimaestro](https://experimaestro-python.readthedocs.io/) services
to **monitor ML learning curves** during experiments. It launches a visualization
backend and aggregates the run directories produced by your tasks so you can watch
training live.

It is intentionally torch-free: the package only depends on `experimaestro`, with
the heavier visualization tools pulled in as optional extras.

## Backends

- **TensorBoard** (`xpm_mlboard.TensorboardService`) — runs `tensorboard` as an
  isolated subprocess on a free port and symlinks each task's tagged run directory
  into a single `runs/` folder.

The `xpm_mlboard.MonitoringService` / `xpm_mlboard.SymlinkMonitoringService`
base classes make it straightforward to add other backends (e.g. Weights & Biases).

## Installation

As a project dependency (with the TensorBoard backend):

```bash
uv add "xpm-mlboard[tensorboard]"
```

### Adding the plugin to `experimaestro` installed as a uv tool

If you run `experimaestro` as a [uv tool](https://docs.astral.sh/uv/concepts/tools/),
inject this plugin into the tool's environment with `--with` (re-run the install to
add the plugin to the existing tool):

```bash
uv tool install experimaestro --with "xpm-mlboard[tensorboard]"
```

## Usage

Add the service to your experiment and register each task's run directory:

```python
from xpm_mlboard import TensorboardService

# `xp` is the experimaestro experiment
service = xp.add_service(TensorboardService(xp.resultspath / "runs"))

# When you submit a task, register its run directory so it shows up:
task = MyLearningTask(...).submit()
service.add(task, task.logpath)
```

Wiring the service into an experiment is typically done through a project-specific
experiment helper (for instance `xpm_torch.experiments.LearningExperimentHelper`,
which exposes `helper.monitoring_service`).

### Custom backends

Subclass `SymlinkMonitoringService` (filesystem-based backends) or
`MonitoringService` (anything else) and implement the backend-specific bits:

```python
from xpm_mlboard import SymlinkMonitoringService
from experimaestro.scheduler.services import ProcessWebService


class MyBackendService(SymlinkMonitoringService, ProcessWebService):
    id = "mybackend"

    def description(self):
        return "My backend service"

    def _build_command(self):
        ...

    def _wait_for_ready(self):
        ...
```

## License

GPL-3
