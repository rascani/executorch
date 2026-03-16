#![no_std]

pub trait Platform {
    fn current_ticks(&self) -> u64;
    fn ticks_to_ns(&self, ticks: u64) -> u64;
}
