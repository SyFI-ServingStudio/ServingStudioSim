"""Entry-point naming shared by the Neuron registry rows and runners (no torch)."""


def entry_name(kind: str, impl: str) -> str:
    return f"profile_{kind}__{impl}"
