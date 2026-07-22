"""Attention specs — one file per mechanism (GQA/MHA/MQA today; MLA/SWA/SSM future)."""

from .gqa import GQA

__all__ = ["GQA"]
