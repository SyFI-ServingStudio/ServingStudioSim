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
    pub fn from_ms(ms: f64) -> Time {
        Time((ms * 1_000_000.0) as u64)
    }

    #[must_use]
    pub fn from_s(s: f64) -> Time {
        Time((s * 1_000_000_000.0) as u64)
    }

    #[must_use]
    pub const fn as_ns(self) -> u64 {
        self.0
    }

    #[must_use]
    pub fn as_ms(self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }

    #[must_use]
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
