//! This example toggles RP Pico/W GPIO  pin 0--connect a resistor and LED and it will blink..
#![no_std]
#![no_main]

extern crate alloc;

#[global_allocator]
static ALLOCATOR: CortexMHeap = CortexMHeap::empty();
const HEAP_SIZE: usize = 1024 * 64; // in bytes

use alloc_cortex_m::CortexMHeap;
use defmt::*;
use embassy_executor::Spawner;
use embassy_rp::gpio::{self, AnyPin};
use embassy_time::Timer;
use gpio::{Level, Output};
use {defmt_rtt as _, panic_probe as _}; // Adjust the import path according to your setup

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    unsafe { ALLOCATOR.init(cortex_m_rt::heap_start() as usize, HEAP_SIZE) }
    let p = embassy_rp::init(Default::default());

    // set pins 0 to 7 for output
    let mut pins = [
        gpio::Output::new(p.PIN_0, Level::High),
        gpio::Output::new(p.PIN_1, Level::High),
        gpio::Output::new(p.PIN_2, Level::High),
        gpio::Output::new(p.PIN_3, Level::High),
        gpio::Output::new(p.PIN_4, Level::High),
        gpio::Output::new(p.PIN_5, Level::High),
        gpio::Output::new(p.PIN_6, Level::High),
        gpio::Output::new(p.PIN_7, Level::High),
    ];
    set_pin_levels(&mut pins, 0b10101010);
    loop {} // run forever

    // Light up the LED

    // Wait for 1 second

    // Turn off the LED

    // Wait for 1 second

    // Loop forever
}

fn set_pin_levels(pins: &mut [gpio::Output; 8], value: u8) {
    for (i, pin) in pins.iter_mut().enumerate() {
        pin.set_level(match (value >> i) & 1 {
            1 => Level::High,
            _ => Level::Low,
        });
    }
}
