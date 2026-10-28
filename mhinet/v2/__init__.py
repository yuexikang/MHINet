"""Isolated MHI v2 research implementation; legacy entrypoints stay unchanged."""

TASK = "mhi_semidense_v2"
SCHEMA = "mhi.v2-run.v1"

from .pipeline import load_mhi_v2
