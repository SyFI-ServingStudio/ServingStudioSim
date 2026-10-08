"""A torch stand-in exposing only the device queries runner device checks read."""

from types import SimpleNamespace


def fake_cuda_torch(
    capability: tuple[int, int] = (9, 0),
    name: str = "NVIDIA GPU",
    *,
    available: bool = True,
) -> SimpleNamespace:
    """``torch`` with one CUDA device of ``capability`` named ``name``, or none."""
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            current_device=lambda: 0,
            get_device_name=lambda _device=None: name,
            get_device_capability=lambda _device=None: capability,
        )
    )
