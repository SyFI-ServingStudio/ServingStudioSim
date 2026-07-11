"""Build alignment-derived inputs for the generic timing-predict launcher.

This package owns conversion semantics only. The launcher owns phase config
parsing, artifact discovery, validation order, and invoking timing-predict.
"""

from .builder import BuildRequest, BuildResult, VllmTextInputSpec, build_inputs

__all__ = ["BuildRequest", "BuildResult", "VllmTextInputSpec", "build_inputs"]
