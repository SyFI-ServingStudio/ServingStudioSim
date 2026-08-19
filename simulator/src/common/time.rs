//! Simulator clock: nanosecond-precision unsigned monotonic time.

use std::ops::{Add, AddAssign, Sub, SubAssign};

use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Time(pub u64);

impl Time {
    pub const ZERO: Time = Time(0);

    #[must_use]
    pub const fn from_ns(ns: u64) -> Time {
        Time(ns)
    }

    #[must_use]
    pub const fn from_us(us: u64) -> Time {
        Time(us * 1_000)
    }

    #[must_use]
    pub const fn from_ms_u64(ms: u64) -> Time {
        Time(ms * 1_000_000)
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "ms is expected non-negative (a duration/offset); callers pass config-derived \
                  or already-validated values, never an arbitrary signed float, so this does not \
                  silently launder a negative input into a huge unsigned Time"
    )]
    pub fn from_ms(ms: f64) -> Time {
        Time((ms * 1_000_000.0) as u64)
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "s is expected non-negative (a duration/offset); callers pass config-derived \
                  or already-validated values, never an arbitrary signed float, so this does not \
                  silently launder a negative input into a huge unsigned Time"
    )]
    pub fn from_s(s: f64) -> Time {
        Time((s * 1_000_000_000.0) as u64)
    }

    #[must_use]
    pub const fn as_ns(self) -> u64 {
        self.0
    }

    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "f64 exactly represents integer ns up to 2^52 (~52 simulated days); every preset \
                  in this repo caps duration_ms well under an hour, and each call converts the \
                  authoritative u64 ns directly (no chained/accumulated float error across ticks), \
                  so real runs stay far inside exact range — but this is not statically enforced \
                  and a multi-week simulated run would start losing sub-ns precision here"
    )]
    pub fn as_ms(self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }

    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "f64 exactly represents integer ns up to 2^52 (~52 simulated days); every preset \
                  in this repo caps duration_ms well under an hour, and each call converts the \
                  authoritative u64 ns directly (no chained/accumulated float error across ticks), \
                  so real runs stay far inside exact range — but this is not statically enforced \
                  and a multi-week simulated run would start losing sub-ns precision here"
    )]
    pub fn as_s(self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }
}

impl Add for Time {
    type Output = Time;
    fn add(self, rhs: Time) -> Time {
        Time(self.0 + rhs.0)
    }
}

impl AddAssign for Time {
    fn add_assign(&mut self, rhs: Time) {
        self.0 += rhs.0;
    }
}

impl Sub for Time {
    type Output = Time;
    fn sub(self, rhs: Time) -> Time {
        Time(self.0 - rhs.0)
    }
}

impl SubAssign for Time {
    fn sub_assign(&mut self, rhs: Time) {
        self.0 -= rhs.0;
    }
}
