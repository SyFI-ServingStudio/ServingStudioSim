"""Source-exact ragged workload splitting shared by V4 indexer prefill leaves."""

from dataclasses import dataclass

from profiling.runners.exceptions import ProfilerNotImplemented

_MAX_REQUESTS = 64
_PRODUCTION_LOGITS_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class IndexerPrefillChunk:
    pairs: tuple[tuple[int, int], ...]
    query_slice_start: int
    query_slice_stop: int
    row_starts: tuple[int, ...]
    row_ends: tuple[int, ...]

    @property
    def num_queries(self) -> int:
        return self.query_slice_stop - self.query_slice_start

    @property
    def num_keys(self) -> int:
        return sum(context // 4 for _, context in self.pairs)

    @property
    def valid_key_pairs(self) -> int:
        return sum(end - start for start, end in zip(self.row_starts, self.row_ends))


def build_indexer_prefill_chunks(
    query_context_pairs: tuple[tuple[int, int], ...],
    max_model_len: int,
    max_num_batched_tokens: int,
    max_logits_bytes: int,
    compress_ratio: int,
) -> tuple[IndexerPrefillChunk, ...]:
    """Mirror vLLM's request-greedy then query-slice physical launcher."""
    if not query_context_pairs or len(query_context_pairs) > _MAX_REQUESTS:
        raise ProfilerNotImplemented("indexer prefill supports 1..64 requests")
    for pair in query_context_pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(type(value) is not int for value in pair)
        ):
            raise TypeError("query_context_pairs must contain integer (query, context) pairs")
        query_length, context_length = pair
        if query_length <= 0 or context_length < query_length:
            raise ValueError("each pair must satisfy 0 < query <= context")
    total_queries = sum(query_length for query_length, _ in query_context_pairs)
    if type(max_model_len) is not int or not 1 <= max_model_len <= 1_048_576:
        raise ProfilerNotImplemented("max_model_len must be in 1..1048576")
    if max(context_length for _, context_length in query_context_pairs) > max_model_len:
        raise ValueError("context length exceeds max_model_len")
    if (
        type(max_num_batched_tokens) is not int
        or not total_queries <= max_num_batched_tokens <= 32768
    ):
        raise ValueError("max_num_batched_tokens must cover all queries and be <=32768")
    if max_logits_bytes != _PRODUCTION_LOGITS_BYTES:
        raise ProfilerNotImplemented("max_logits_bytes must be 512 MiB")
    if compress_ratio != 4:
        raise ProfilerNotImplemented("indexer prefill is supported only for C4 layers")

    # The serving model allocates 40 * max_model_len / C4 gathered rows.  Use
    # the physical allocation as the safe cap even though the current metadata
    # builder passes its pre-division upper bound to the splitter.
    workspace_keys = 40 * max_model_len // compress_ratio
    maximum_logits_elements = max_logits_bytes // 4
    chunks: list[IndexerPrefillChunk] = []
    request_start = 0
    while request_start < len(query_context_pairs):
        request_stop = request_start
        chunk_queries = 0
        chunk_keys = 0
        while request_stop < len(query_context_pairs):
            query_length, context_length = query_context_pairs[request_stop]
            candidate_queries = chunk_queries + query_length
            candidate_keys = chunk_keys + context_length // compress_ratio
            if (
                candidate_keys <= workspace_keys
                and candidate_queries * candidate_keys <= maximum_logits_elements
            ):
                chunk_queries = candidate_queries
                chunk_keys = candidate_keys
                request_stop += 1
            else:
                break
        if request_stop == request_start:
            query_length, context_length = query_context_pairs[request_stop]
            chunk_queries = query_length
            chunk_keys = context_length // compress_ratio
            request_stop += 1
        if chunk_keys <= 0:
            request_start = request_stop
            continue
        if chunk_keys > workspace_keys:
            raise ProfilerNotImplemented("one request exceeds the physical gather workspace")

        pairs = query_context_pairs[request_start:request_stop]
        maximum_queries = max(1, maximum_logits_elements // chunk_keys)
        for query_slice_start in range(0, chunk_queries, maximum_queries):
            query_slice_stop = min(query_slice_start + maximum_queries, chunk_queries)
            row_starts, row_ends = _row_spans(
                pairs, query_slice_start, query_slice_stop, compress_ratio
            )
            chunks.append(
                IndexerPrefillChunk(
                    pairs,
                    query_slice_start,
                    query_slice_stop,
                    row_starts,
                    row_ends,
                )
            )
        request_start = request_stop
    if not chunks:
        raise ProfilerNotImplemented("workload has no compressed C4 keys")
    return tuple(chunks)


def _row_spans(
    pairs: tuple[tuple[int, int], ...],
    query_slice_start: int,
    query_slice_stop: int,
    compress_ratio: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    starts: list[int] = []
    ends: list[int] = []
    request_key_base = 0
    for query_length, context_length in pairs:
        prefix_length = context_length - query_length
        for query_offset in range(query_length):
            starts.append(request_key_base)
            ends.append(request_key_base + (prefix_length + query_offset + 1) // compress_ratio)
        request_key_base += context_length // compress_ratio
    return (
        tuple(starts[query_slice_start:query_slice_stop]),
        tuple(ends[query_slice_start:query_slice_stop]),
    )


__all__ = ["IndexerPrefillChunk", "build_indexer_prefill_chunks"]
