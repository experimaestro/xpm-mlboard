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
- **Weights & Biases** — *in development.* See [Weights & Biases](#weights--biases).

The `xpm_mlboard.MonitoringService` / `xpm_mlboard.SymlinkMonitoringService`
base classes make it straightforward to add other backends (e.g. Weights & Biases).
Alongside the symlink aggregate, `SymlinkMonitoringService` records each run's
experimaestro tags in a `<tagspath>.tags.json` sidecar (recoverable with
`xpm_mlboard.service.read_run_tags`), so backend-agnostic exporters can attach
the right tags to each run.

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

## Weights & Biases

> **Status:** the exporter is available as a library
> (`xpm_mlboard.exporters.wandb.sync`). Triggering it from the experimaestro
> TUI / Web UI and CLI, and a live "sync" service, are still in development. The
> `wandb` configuration below is standard and applies to all of these.

### How it works

Each experimaestro **experiment** maps to a W&B **project**, and every task
becomes a **run** named after its tags, with the task's experimaestro tags
attached as the run's W&B `config`/`tags`. Because compute nodes are frequently
offline, tasks never talk to W&B themselves — they only write TensorBoard event
files. The driver-side exporter reads those event files together with the
`<tagspath>.tags.json` sidecars written by the monitoring service and pushes them
to W&B. Each run gets a **deterministic id** derived from its tags, so re-running
the export (or syncing an offline run later) updates the same run instead of
creating duplicates. The same shared task can therefore be exported into several
experiments, each carrying that experiment's own tags.

### Running the export

Install the extra and call `sync()` on a monitoring aggregate directory (the
service's `runs/` folder):

```bash
uv add "xpm-mlboard[wandb]"
```

```python
from xpm_mlboard.exporters.wandb import sync

# Export every run under <results>/runs into the "my-experiment" W&B project.
sync(runs_dir, project="my-experiment")

# Also remove W&B runs whose task is no longer part of the experiment:
sync(runs_dir, project="my-experiment", reconcile=True)
```

`sync()` also accepts `mode` (`online` / `offline` / `disabled`; otherwise taken
from `WANDB_MODE`) and `entity`. Make sure W&B access is configured first (see
below).

### Configuring access

W&B authenticates with an API key. Either log in once (stored in `~/.netrc`):

```bash
wandb login                     # prompts for your API key
```

or provide it via the environment (handy on CI / headless driver hosts):

```bash
export WANDB_API_KEY=...
```

### Using a self-hosted W&B server

Point the client at your own W&B Server ("Local") instead of `wandb.ai`:

```bash
wandb login --host https://wandb.your-domain.example      # interactive
# or non-interactively:
export WANDB_BASE_URL=https://wandb.your-domain.example
export WANDB_API_KEY=...
```

### Offline / air-gapped hosts

When the host running the export has no network at sync time (e.g. a SLURM login
node), write run data locally and upload it later from a connected machine:

```bash
export WANDB_MODE=offline        # record locally instead of streaming
# ... later, from a host that can reach the server:
wandb sync <run-dir>
```

Common environment variables: `WANDB_API_KEY`, `WANDB_BASE_URL` (self-hosted
server URL), `WANDB_MODE` (`online` / `offline` / `disabled`), `WANDB_ENTITY`
(team or user), `WANDB_PROJECT`. See the [W&B docs](https://docs.wandb.ai/) for
the complete reference.

## License

GPL-3
