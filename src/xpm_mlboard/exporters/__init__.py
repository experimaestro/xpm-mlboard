"""Driver-side exporters that turn a monitoring aggregate (TensorBoard event
files + tag sidecars) into another backend's runs.

Exporters are intentionally kept out of the monitoring path: a task only writes
TensorBoard event files (which works offline), and an exporter reads those —
together with the ``<tagspath>.tags.json`` sidecars written by
:class:`xpm_mlboard.SymlinkMonitoringService` — to populate the target backend.
"""
