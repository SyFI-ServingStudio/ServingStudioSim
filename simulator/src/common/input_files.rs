//! Input-file reads with an in-memory overlay.
//!
//! Model configs and routing artifacts are named by path in an arch block. A
//! host with no filesystem (the wasm32 build) installs their contents with
//! [`with_files`] instead; a read of a path the overlay does not hold falls
//! back to `std::fs`, so native callers see no change.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

thread_local! {
    static OVERLAY: RefCell<HashMap<PathBuf, String>> = RefCell::new(HashMap::new());
}

/// The overlay's text for `path`, else the file's.
pub fn read_to_string(path: &Path) -> std::io::Result<String> {
    if let Some(text) = OVERLAY.with(|o| o.borrow().get(path).cloned()) {
        return Ok(text);
    }
    std::fs::read_to_string(path)
}

/// Run `f` with `files` (path -> contents) readable through [`read_to_string`],
/// then restore the previous overlay.
pub fn with_files<R>(files: HashMap<PathBuf, String>, f: impl FnOnce() -> R) -> R {
    let previous = OVERLAY.with(|o| std::mem::replace(&mut *o.borrow_mut(), files));
    struct Restore(Option<HashMap<PathBuf, String>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            if let Some(previous) = self.0.take() {
                OVERLAY.with(|o| *o.borrow_mut() = previous);
            }
        }
    }
    let _restore = Restore(Some(previous));
    f()
}
