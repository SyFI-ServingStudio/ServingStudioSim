//! Prefill token-budget gate for local admission lifecycles.

/// Reserve decode tokens first, then fill the remaining per-iteration budget
/// with whole prefills. One over-budget head request is allowed when the
/// partition has non-zero prefill budget and admitted no prefill yet, preventing
/// starvation. KV capacity remains a separate `KvStore::fits` decision.
pub(crate) fn prefill_fits_budget(
    budget: u32,
    decode_tokens: u32,
    admitted_tokens: u32,
    next_prefill_tokens: u32,
) -> bool {
    let prefill_budget = budget.saturating_sub(decode_tokens);
    let fits = admitted_tokens.saturating_add(next_prefill_tokens) <= prefill_budget;
    let force_first =
        admitted_tokens == 0 && prefill_budget > 0 && next_prefill_tokens > prefill_budget;
    fits || force_first
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_until_budget_is_exhausted() {
        assert!(prefill_fits_budget(20, 0, 0, 8));
        assert!(prefill_fits_budget(20, 0, 8, 8));
        assert!(!prefill_fits_budget(20, 0, 16, 8));
    }

    #[test]
    fn reserves_decode_tokens_before_prefill() {
        assert!(prefill_fits_budget(12, 10, 0, 2));
        assert!(!prefill_fits_budget(12, 10, 2, 1));
    }

    #[test]
    fn admits_no_prefill_when_decode_consumes_budget() {
        assert!(!prefill_fits_budget(8, 8, 0, 1));
        assert!(!prefill_fits_budget(8, 20, 0, 1));
    }

    #[test]
    fn force_admits_only_the_first_overlong_prefill() {
        assert!(prefill_fits_budget(4, 0, 0, 10));
        assert!(!prefill_fits_budget(4, 0, 10, 10));
    }
}
