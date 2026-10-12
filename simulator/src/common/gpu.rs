//! GPU facts from the repository's GPU catalog, `gpu/spec.json`, the same
//! table the Python profiler reads (`profiling.gpu_catalog`).

use std::collections::HashMap;
use std::sync::OnceLock;

/// The CUDA compute capability `(major, minor)` of a GPU, looked up by its
/// catalog name or any alias (e.g. "NVIDIA B200"), trimmed and ignoring ASCII
/// case, first catalog entry first, as `profiling.gpu_catalog.resolve_gpu_spec`
/// and the Analyzer's `hardware::resolve_gpu` match. `None` for a GPU the catalog does not list or a non-NVIDIA part.
pub fn compute_capability(gpu_name: &str) -> Option<(u32, u32)> {
    static BY_NAME: OnceLock<HashMap<String, (u32, u32)>> = OnceLock::new();
    BY_NAME.get_or_init(load).get(&key(gpu_name)).copied()
}

/// The catalog name of a GPU looked up by its name or any alias, matched as
/// [`compute_capability`] matches. `None` for a GPU the catalog does not list.
pub fn canonical_name(gpu_name: &str) -> Option<&'static str> {
    static BY_NAME: OnceLock<HashMap<String, &'static str>> = OnceLock::new();
    BY_NAME
        .get_or_init(|| {
            let mut by_name = HashMap::new();
            for gpu in catalog() {
                let canonical: &'static str = Box::leak(gpu.name.clone().into_boxed_str());
                for name in std::iter::once(&gpu.name).chain(&gpu.aliases) {
                    by_name.entry(key(name)).or_insert(canonical);
                }
            }
            by_name
        })
        .get(&key(gpu_name))
        .copied()
}

/// The form a GPU name is matched in.
fn key(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}

#[derive(serde::Deserialize)]
struct Gpu {
    name: String,
    #[serde(default)]
    aliases: Vec<String>,
    compute_capability: Option<String>,
}

fn catalog() -> Vec<Gpu> {
    #[derive(serde::Deserialize)]
    struct Catalog {
        gpus: Vec<Gpu>,
    }

    let catalog: Catalog = serde_json::from_str(include_str!("../../../gpu/spec.json"))
        .expect("gpu/spec.json is a GPU catalog");
    catalog.gpus
}

fn load() -> HashMap<String, (u32, u32)> {
    let mut by_name = HashMap::new();
    for gpu in catalog() {
        let Some(text) = gpu.compute_capability else {
            continue;
        };
        let capability = text
            .split_once('.')
            .and_then(|(major, minor)| Some((major.parse().ok()?, minor.parse().ok()?)))
            .unwrap_or_else(|| {
                panic!(
                    "{}: compute_capability {text:?} is not major.minor",
                    gpu.name
                )
            });
        for name in std::iter::once(gpu.name).chain(gpu.aliases) {
            by_name.entry(key(&name)).or_insert(capability);
        }
    }
    by_name
}

#[cfg(test)]
mod tests {
    use super::{canonical_name, compute_capability};

    #[test]
    fn aliases_resolve_to_the_catalog_name() {
        assert_eq!(canonical_name("AWS Trainium2 LNC2"), Some("Trainium2-LNC2"));
        assert_eq!(canonical_name(" trainium2-lnc2"), Some("Trainium2-LNC2"));
        assert_eq!(canonical_name("No Such GPU"), None);
        assert_eq!(compute_capability("AWS Trainium2 LNC2"), None);
    }

    #[test]
    fn names_and_aliases_resolve_to_the_catalog_capability() {
        assert_eq!(compute_capability("NVIDIA H200"), Some((9, 0)));
        assert_eq!(compute_capability("B200-SXM-180GB"), Some((10, 0)));
        assert_eq!(compute_capability("NVIDIA B300"), Some((10, 3)));
        assert_eq!(compute_capability("  nvidia h200 "), Some((9, 0)));
        assert_eq!(compute_capability("AMD MI300X"), None);
        assert_eq!(compute_capability("No Such GPU"), None);
    }
}
