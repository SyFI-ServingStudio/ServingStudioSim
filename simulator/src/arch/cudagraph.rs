//! vLLM CUDA-graph dispatch rules shared by the vLLM model archs: the row
//! count a step replays at, and which uniform-decode steps replay a FULL graph
//! (`vllm/v1/cudagraph_dispatcher.py`).

/// The row count vLLM runs a CUDA-graph replay on: the smallest captured size
/// that holds `tokens`, or `tokens` itself above the largest (eager).
pub fn graph_padded_tokens(sorted_sizes: &[u32], tokens: u32) -> u32 {
    match sorted_sizes.binary_search(&tokens) {
        Ok(_) => tokens,
        Err(index) => sorted_sizes.get(index).copied().unwrap_or(tokens),
    }
}

/// `--cudagraph-capture-sizes` as vLLM holds them: ascending, deduplicated.
pub fn sorted_capture_sizes(sizes: &[u32]) -> Vec<u32> {
    let mut sizes = sizes.to_vec();
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

/// The largest one-token-per-request decode step vLLM replays as a FULL CUDA
/// graph under FULL_AND_PIECEWISE, or 0 when none is captured. vLLM keeps a
/// FULL decode key only for a captured size of at most `max_num_seqs` tokens
/// (`CudagraphDispatcher.initialize_cudagraph_keys`), so a decode step that
/// pads past this bound replays a PIECEWISE graph, and one past the largest
/// size runs eagerly.
pub fn full_decode_graph_max_tokens(sorted_sizes: &[u32], max_num_seqs: u32) -> u32 {
    let within = sorted_sizes.partition_point(|&size| size <= max_num_seqs);
    within.checked_sub(1).map_or(0, |last| sorted_sizes[last])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_padding_rounds_up_to_the_next_captured_size() {
        let sizes = [1, 2, 4, 8, 16, 24];
        let padded: Vec<u32> = [1, 3, 5, 16, 17, 24, 25]
            .iter()
            .map(|&t| graph_padded_tokens(&sizes, t))
            .collect();
        assert_eq!(padded, [1, 4, 8, 16, 24, 24, 25]);
        assert_eq!(graph_padded_tokens(&[], 7), 7);
    }

    #[test]
    fn full_decode_graphs_stop_at_the_largest_size_within_max_num_seqs() {
        // An explicit 2048 cap captures no 100: 97-100 requests pad to 104,
        // which has no FULL decode key at max_num_seqs 100.
        let sizes = [1, 2, 4, 8, 96, 104, 2048];
        assert_eq!(full_decode_graph_max_tokens(&sizes, 100), 96);
        assert_eq!(full_decode_graph_max_tokens(&sizes, 104), 104);
        assert_eq!(full_decode_graph_max_tokens(&sizes, 4096), 2048);
        assert_eq!(full_decode_graph_max_tokens(&sizes[1..], 1), 0);
        assert_eq!(full_decode_graph_max_tokens(&[], 100), 0);
    }
}
