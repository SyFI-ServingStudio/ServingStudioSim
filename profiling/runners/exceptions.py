"""Typed runner failures understood by L1b."""

from __future__ import annotations


class OOMError(RuntimeError):
    """The profiled shape ran out of device memory."""


class KernelLaunchFailed(RuntimeError):
    """The backend failed while launching or synchronizing the measured kernel."""


class ProfilerNotImplemented(RuntimeError):
    """The requested profiler/backend exists in schema but is not runnable here."""
