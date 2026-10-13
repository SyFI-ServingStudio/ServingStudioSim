"""Independent PyTorch reference for weight-free ReLU-sum indexer scores."""

from typing import Any


def sparse_indexer_logits_reference(
    torch: Any,
    q: Any,
    k_cache: Any,
    page_table_row: Any,
    num_keys: int,
) -> Any:
    """Scores ``[rows, num_keys]`` of query rows ``q`` ``[rows, heads, dim]`` against
    the first ``num_keys`` keys of one request's pages (``k_cache`` is
    ``[pages, page_size, 1, dim]``)."""
    page_size = k_cache.shape[1]
    columns = torch.arange(num_keys, device=q.device)
    pages = page_table_row[columns // page_size].long()
    keys = k_cache[pages, columns % page_size, 0].float()  # [keys, dim]
    dots = torch.einsum("rhd,kd->rhk", q.float(), keys)
    return dots.clamp_min(0).sum(dim=1)


__all__ = ["sparse_indexer_logits_reference"]
