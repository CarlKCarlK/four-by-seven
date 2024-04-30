#![no_std]
#![no_main]

extern crate alloc;

#[global_allocator]
static ALLOCATOR: CortexMHeap = CortexMHeap::empty();
const HEAP_SIZE: usize = 1024 * 64; // in bytes

use core::array;
use core::cmp::max;
use heapless::{LinearMap, Vec};

use alloc_cortex_m::CortexMHeap;
use defmt::unwrap;
use embassy_executor::{Executor, Spawner};
use embassy_futures::select::{select, Either};
use embassy_rp::adc::{Adc, Async, Channel as AdcChannel, Config, InterruptHandler};
use embassy_rp::{
    bind_interrupts, gpio,
    multicore::{spawn_core1, Stack},
    peripherals::CORE1,
};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::channel::Channel;
use embassy_sync::mutex::Mutex;
use embassy_time::{Duration, Timer};
use embedded_hal_1::digital::OutputPin;
use gpio::Level;
use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};
use static_cell::StaticCell;
use {defmt_rtt as _, panic_probe as _}; // Adjust the import path according to your setup

bind_interrupts!(struct Irqs {
    ADC_IRQ_FIFO => InterruptHandler;
});
pub struct VirtualPotentiometer {
    mutex_level: Mutex<CriticalSectionRawMutex, u16>,
}

impl VirtualPotentiometer {
    async fn read(&'static self) -> u16 {
        let level = self.mutex_level.lock().await;
        *level
    }
    async fn multiplex(
        &'static self,
        acd: &'static mut Adc<'static, Async>,
        adc_pin: &'static mut AdcChannel<'static>,
    ) {
        loop {
            let level_in = acd.read(adc_pin).await;
            if let Ok(level_in) = level_in {
                let mut level_out = self.mutex_level.lock().await;
                *level_out = level_in;
            }
            Timer::after(Duration::from_millis(100)).await;
        }
    }
}

pub struct VirtualDisplay<const DIGIT_COUNT: usize> {
    mutex_digits: Mutex<CriticalSectionRawMutex, [u8; DIGIT_COUNT]>,
    update_display_channel: Channel<CriticalSectionRawMutex, (), 1>,
}

impl<const DIGIT_COUNT: usize> VirtualDisplay<DIGIT_COUNT> {
    pub async fn write_text(&'static self, text: &str) {
        let bytes = line_to_u8_array(text);
        self.write_bytes(bytes).await;
    }
    pub async fn write_bytes(&'static self, bytes_in: [u8; DIGIT_COUNT]) {
        {
            // inner scope to release the lock
            let mut bytes_out = self.mutex_digits.lock().await;
            for (byte_out, byte_in) in bytes_out.iter_mut().zip(bytes_in.iter()) {
                *byte_out = *byte_in;
            }
        }
        // Say that the display should be updated. If a previous update is
        // still pending, this new update can be ignored.
        let _ = self.update_display_channel.try_send(());
    }

    pub async fn write_number(&'static self, mut number: u16) {
        let mut bytes = [0; DIGIT_COUNT];

        for i in (0..DIGIT_COUNT).rev() {
            let digit = (number % 10) as usize; // Get the last digit
            bytes[i] = Leds::DIGITS[digit];
            number /= 10; // Remove the last digit
            if number == 0 {
                break;
            }
        }

        // If the original number was out of range, turn on all decimal points
        if number > 0 {
            for byte in bytes.iter_mut() {
                *byte |= Leds::DECIMAL;
            }
        }
        self.write_bytes(bytes).await;
    }

    #[allow(clippy::needless_range_loop)]
    async fn multiplex(
        &'static self,
        digit_pins: &'static mut [gpio::Output<'_>; DIGIT_COUNT],
        segment_pins: &'static mut [gpio::Output<'_>; 8],
    ) {
        loop {
            // How many unique, non-blank digits?
            let mut map: LinearMap<u8, Vec<usize, DIGIT_COUNT>, DIGIT_COUNT> = LinearMap::new();
            {
                // inner scope to release the lock
                let digits = self.mutex_digits.lock().await;
                let digits = digits.iter();
                for (index, byte) in digits.enumerate() {
                    if *byte != 0 {
                        if let Some(vec) = map.get_mut(byte) {
                            vec.push(index).unwrap();
                        } else {
                            let mut vec = Vec::default();
                            vec.push(index).unwrap();
                            map.insert(*byte, vec).unwrap();
                        }
                    }
                }
            }
            match map.len() {
                // If the display should be empty, then just wait for the next update
                0 => self.update_display_channel.receive().await,
                // If only one pattern should be displayed (even on multiple digits), display it
                // and wait for the next update
                1 => {
                    // get one and only key and value
                    let (byte, indexes) = map.iter().next().unwrap();
                    // Set the segment pins with the bool iterator
                    bool_iter(*byte).zip(segment_pins.iter_mut()).for_each(
                        |(state, segment_pin)| {
                            segment_pin.set_state(state.into()).unwrap();
                        },
                    );
                    // activate the digits, wait for the next update, and deactivate the digits
                    for digit_index in indexes.iter() {
                        digit_pins[*digit_index].set_low(); // Assuming common cathode setup
                    }
                    self.update_display_channel.receive().await;
                    for digit_index in indexes.iter() {
                        digit_pins[*digit_index].set_high();
                    }
                }
                // If multiple patterns should be displayed, multiplex them until the next update
                _ => {
                    loop {
                        for (byte, indexes) in map.iter() {
                            // Set the segment pins with the bool iterator
                            bool_iter(*byte).zip(segment_pins.iter_mut()).for_each(
                                |(state, segment_pin)| {
                                    segment_pin.set_state(state.into()).unwrap();
                                },
                            );
                            // Activate, pause, and deactivate the digits
                            for digit_index in indexes.iter() {
                                digit_pins[*digit_index].set_low(); // Assuming common cathode setup
                            }
                            // cmk improve overflow, scaling, avoiding 1, etc.
                            let mut sleep = scale_adc_value(VIRTUAL_POTENTIOMETER1.read().await);
                            sleep = sleep * DIGIT_COUNT as u64 / map.len() as u64;
                            // Sleep (but wake up early if the display should be updated)
                            select(
                                Timer::after(Duration::from_millis(sleep)),
                                self.update_display_channel.receive(),
                            )
                            .await;
                            for digit_index in indexes.iter() {
                                digit_pins[*digit_index].set_high();
                            }
                        }
                        // break out of multiplexing loop if the display should be updated
                        if self.update_display_channel.try_receive().is_err() {
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Turn a u8 into an iterator of bool
    pub async fn bool_iter(&'static self, digit_index: usize) -> array::IntoIter<bool, 8> {
        // inner scope to release the lock
        let byte: u8;
        {
            let digit_array = self.mutex_digits.lock().await;
            byte = digit_array[digit_index];
        }
        bool_iter(byte)
    }
}

#[inline]
/// Turn a u8 into an iterator of bool
pub fn bool_iter(mut byte: u8) -> array::IntoIter<bool, 8> {
    // turn a u8 into an iterator of bool
    let mut bools_out = [false; 8];
    for bool_out in bools_out.iter_mut() {
        *bool_out = byte & 1 == 1;
        byte >>= 1;
    }
    bools_out.into_iter()
}

// Display #1 is a 4-digit 7-segment display
pub const DIGIT_COUNT1: usize = 4;

static VIRTUAL_DISPLAY1: VirtualDisplay<DIGIT_COUNT1> = VirtualDisplay {
    mutex_digits: Mutex::new([255; DIGIT_COUNT1]),
    update_display_channel: Channel::new(),
};

#[embassy_executor::task]
async fn multiplex_display1(
    digit_pins: &'static mut [gpio::Output<'_>; DIGIT_COUNT1],
    segment_pins: &'static mut [gpio::Output<'_>; 8],
) {
    VIRTUAL_DISPLAY1.multiplex(digit_pins, segment_pins).await;
}

static VIRTUAL_POTENTIOMETER1: VirtualPotentiometer = VirtualPotentiometer {
    mutex_level: Mutex::new(12),
};

#[embassy_executor::task]
async fn multiplex_potentiometer1(
    adc: &'static mut Adc<'static, Async>,
    adc_pin: &'static mut AdcChannel<'static>,
) {
    VIRTUAL_POTENTIOMETER1.multiplex(adc, adc_pin).await;
}

struct Pins {
    digits1: &'static mut [gpio::Output<'static>; DIGIT_COUNT1],
    segments1: &'static mut [gpio::Output<'static>; 8],
    button: &'static mut gpio::Input<'static>,
    led0: &'static mut gpio::Output<'static>,
    adc: &'static mut Adc<'static, Async>,
    adc_pin: &'static mut AdcChannel<'static>,
}

impl Pins {
    fn new() -> (Self, CORE1) {
        let p: embassy_rp::Peripherals = embassy_rp::init(Default::default());
        let core1 = p.CORE1;

        static DIGIT_PINS1: StaticCell<[gpio::Output; DIGIT_COUNT1]> = StaticCell::new();
        let digits1 = DIGIT_PINS1.init([
            gpio::Output::new(p.PIN_1, Level::High),
            gpio::Output::new(p.PIN_2, Level::High),
            gpio::Output::new(p.PIN_3, Level::High),
            gpio::Output::new(p.PIN_4, Level::High),
        ]);

        static SEGMENT_PINS1: StaticCell<[gpio::Output; 8]> = StaticCell::new();
        let segments1 = SEGMENT_PINS1.init([
            gpio::Output::new(p.PIN_5, Level::Low),
            gpio::Output::new(p.PIN_6, Level::Low),
            gpio::Output::new(p.PIN_7, Level::Low),
            gpio::Output::new(p.PIN_8, Level::Low),
            gpio::Output::new(p.PIN_9, Level::Low),
            gpio::Output::new(p.PIN_10, Level::Low),
            gpio::Output::new(p.PIN_11, Level::Low),
            gpio::Output::new(p.PIN_12, Level::Low),
        ]);

        static BUTTON_PIN: StaticCell<gpio::Input> = StaticCell::new();
        let button = BUTTON_PIN.init(gpio::Input::new(p.PIN_13, gpio::Pull::Down));

        static LED0_PIN: StaticCell<gpio::Output> = StaticCell::new();
        let led0 = LED0_PIN.init(gpio::Output::new(p.PIN_0, Level::Low));

        static ADC: StaticCell<Adc<Async>> = StaticCell::new();
        let adc = ADC.init(Adc::new(p.ADC, Irqs, Config::default()));
        static ADC_PIN: StaticCell<AdcChannel> = StaticCell::new();
        let adc_pin = ADC_PIN.init(AdcChannel::new_pin(p.PIN_26, gpio::Pull::None));

        (
            Self {
                digits1,
                segments1,
                button,
                led0,
                adc,
                adc_pin,
            },
            core1,
        )
    }
}

static mut CORE1_STACK: Stack<4096> = Stack::new();
static EXECUTOR1: StaticCell<Executor> = StaticCell::new();

fn scale_adc_value(adc_value: u16) -> u64 {
    let old_min: u16 = 15;
    let old_max: u16 = 4076;
    let new_min: u64 = 1;
    let new_max: u64 = 100;

    // Use saturating_sub to prevent underflow
    let safe_adc_value = adc_value.saturating_sub(old_min);

    // Casts are necessary to ensure proper division and multiplication
    let scaled_value =
        (safe_adc_value as u64 * (new_max - new_min)) / (old_max as u64 - old_min as u64) + new_min;

    // Ensure the value never goes below 1 (though it should not by calculation)
    max(scaled_value, new_min)
}
#[embassy_executor::main]
async fn main(_spawner0: Spawner) {
    unsafe { ALLOCATOR.init(cortex_m_rt::heap_start() as usize, HEAP_SIZE) }

    let (pins, core1) = Pins::new();

    // Spawn 'multiplex_display1' on core1
    spawn_core1(
        core1,
        unsafe { &mut *core::ptr::addr_of_mut!(CORE1_STACK) },
        move || {
            let executor1 = EXECUTOR1.init(Executor::new());
            executor1.run(|spawner1| {
                unwrap!(spawner1.spawn(multiplex_display1(pins.digits1, pins.segments1)));
                unwrap!(spawner1.spawn(multiplex_potentiometer1(pins.adc, pins.adc_pin)));
            });
        },
    );

    // Display "RUST" on the 4-digit 7-segment display while we render the movies
    VIRTUAL_DISPLAY1.write_text("RUST").await;

    // Render the movies -- this is CPU intensive and will run on core0
    let render_movies: [RangeMapBlaze<i32, [u8; DIGIT_COUNT1]>; 2] =
        [hello_world_wide(), circles_wide()];

    // loop through the movies, forever
    for movie in render_movies.iter().cycle() {
        // Loop through the frames of the current movie
        for range_values in movie.range_values() {
            // Get the next frame of the movie (and its duration)
            let (start, end) = range_values.range.into_inner();
            let frame = *range_values.value;

            // Display the frame
            VIRTUAL_DISPLAY1.write_bytes(frame).await;

            // Find the duration that this frame should be displayed
            let frame_count = (end + 1 - start) as u64;
            let duration = Duration::from_millis(frame_count * 1000 / FPS as u64);

            // Wait for the frame to finish or the button to be pressed
            if let Either::First(()) =
                select(Timer::after(duration), pins.button.wait_for_rising_edge()).await
            {
                continue; // Frame finished, so go to the next frame
            }

            // The button was pressed, so wait for it to be released
            pins.led0.set_high(); // mirror button press on led0
            Timer::after(Duration::from_millis(5)).await; // debounce button
            pins.button.wait_for_falling_edge().await;
            pins.led0.set_low();
        }
    }
}

const FPS: i32 = 24;

fn line_to_u8_array<const DIGIT_COUNT: usize>(line: &str) -> [u8; DIGIT_COUNT] {
    let mut result = [0; DIGIT_COUNT];
    (0..DIGIT_COUNT).zip(line.chars()).for_each(|(i, c)| {
        result[i] = Leds::ASCII_TABLE[c as usize];
    });
    if line.len() > DIGIT_COUNT {
        for byte in result.iter_mut() {
            *byte |= Leds::DECIMAL;
        }
    }
    result
}

pub fn hello_world_wide() -> RangeMapBlaze<i32, [u8; DIGIT_COUNT1]> {
    let message =
        "3\n 2\n  1\n\n   H\n  He\n Hel\nHell\nello\nllo\nlo R\no Ru\n Rus\nRust\nust\nst\nt\n";
    let message: RangeMapBlaze<i32, _> = message
        .lines()
        .enumerate()
        .map(|(i, line)| (i as i32, line_to_u8_array(line)))
        .collect();
    let message = linear_wide(&message, FPS, 0);
    // add gaps of 3 frames between each character
    let message = message
        .range_values()
        .enumerate()
        .map(|(i, range_value)| {
            let (start, end) = range_value.range.clone().into_inner();
            let new_range = start + i as i32 * 3..=end + i as i32 * 3;
            (new_range, range_value.value)
        })
        .collect();
    message
}

pub fn hello_world() -> RangeMapBlaze<i32, u8> {
    let message = "321 Hello world!";
    let message: RangeMapBlaze<i32, u8> = message
        .chars()
        .enumerate()
        .map(|(i, c)| (i as i32, Leds::ASCII_TABLE[c as usize]))
        .collect();
    let message = linear(&message, FPS, 0);
    // add gaps of 3 frames between each character
    let message = message
        .range_values()
        .enumerate()
        .map(|(i, range_value)| {
            let (start, end) = range_value.range.clone().into_inner();
            let new_range = start + i as i32 * 3..=end + i as i32 * 3;
            (new_range, range_value.value)
        })
        .collect();
    message
}

pub fn circles_wide() -> RangeMapBlaze<i32, [u8; DIGIT_COUNT1]> {
    // Light up segments A to F
    let circle = RangeMapBlaze::from_iter([
        (0, [0, 0, 0, Leds::SEG_A]),
        (1, [0, 0, 0, Leds::SEG_B]),
        (2, [0, 0, 0, Leds::SEG_C]),
        (3, [0, 0, 0, Leds::SEG_D]),
        (4, [0, 0, Leds::SEG_D, 0]),
        (5, [0, Leds::SEG_D, 0, 0]),
        (6, [Leds::SEG_D, 0, 0, 0]),
        (7, [Leds::SEG_E, 0, 0, 0]),
        (8, [Leds::SEG_F, 0, 0, 0]),
        (9, [Leds::SEG_A, 0, 0, 0]),
        (10, [0, Leds::SEG_A, 0, 0]),
        (11, [0, 0, Leds::SEG_A, 0]),
    ]);
    let mut main = RangeMapBlaze::new();
    let mut scale = 1;
    while scale < 24 {
        // Slow down the circle by a factor of 1 to 24, appending to `main` each time.
        main = &main | linear_wide(&circle, scale, main.len() as i32);
        scale *= 2;
    }
    // append main with itself, but reversed
    main = &main | linear_wide(&main, -1, main.len() as i32);

    // append 10 copies of the fast circle
    for _ in 0..20 {
        main = &main | linear_wide(&circle, -1, main.len() as i32);
    }
    main
}

pub fn circles() -> RangeMapBlaze<i32, u8> {
    // Light up segments A to F
    let circle = RangeMapBlaze::from_iter([
        (0, Leds::SEG_A),
        (1, Leds::SEG_B),
        (2, Leds::SEG_C),
        (3, Leds::SEG_D),
        (4, Leds::SEG_E),
        (5, Leds::SEG_F),
    ]);
    let mut main = RangeMapBlaze::new();
    let mut scale = 1;
    while scale < 24 {
        // Slow down the circle by a factor of 1 to 24, appending to `main` each time.
        main = &main | linear(&circle, scale, main.len() as i32);
        scale *= 2;
    }
    // append main with itself, but reversed
    main = &main | linear(&main, -1, main.len() as i32);

    // append 10 copies of the fast circle
    for _ in 0..20 {
        main = &main | linear(&circle, -1, main.len() as i32);
    }

    main
}

// i32 means we can only go 3 weeks at a time at 24fps. Be sure the code checks this.
pub fn double_count_down() -> RangeMapBlaze<i32, u8> {
    let length_seconds = 30;
    let frame_count = FPS * length_seconds;

    // The `main`` track starts with 15 seconds of black
    let mut main = RangeMapBlaze::from_iter([(0..=frame_count - 1, Leds::SPACE)]);
    // println!("main {main:?}");

    // Create a 10 frame `digits` track with "0" to "9"".
    let mut digits =
        RangeMapBlaze::from_iter((0i32..=9).map(|i| (i..=i, Leds::DIGITS[i as usize])));

    // Make frame 0 be the middle LED segment.
    digits.insert(0, Leds::SEG_G);

    // Oops, we've changed our mind and now don't want frames 8 and 9.
    digits = digits - RangeSetBlaze::from_iter([8..=9]);

    // Apply the following linear transformation to `digits``:
    // 1. Make each original frame last one second
    // 2. Reverse the order of the frames
    // 3. Shift the frames 1 second into the future
    digits = linear(&digits, -FPS, FPS);
    // println!("digits m {digits:?}");

    // Composite these together (listed from top to bottom)
    //  1. `digits``
    //  2. `digits` shifted 10 seconds into the future
    //  3. `main`
    main = &digits | &linear(&digits, 1, 10 * FPS) | &main;
    main
}

pub struct Leds;

#[allow(dead_code)]
impl Leds {
    const SEG_A: u8 = 0b00000001;
    const SEG_B: u8 = 0b00000010;
    const SEG_C: u8 = 0b00000100;
    const SEG_D: u8 = 0b00001000;
    const SEG_E: u8 = 0b00010000;
    const SEG_F: u8 = 0b00100000;
    const SEG_G: u8 = 0b01000000;
    const DECIMAL: u8 = 0b10000000;

    const DIGITS: [u8; 10] = [
        0b00111111, // Digit 0
        0b00000110, // Digit 1
        0b01011011, // Digit 2
        0b01001111, // Digit 3
        0b01100110, // Digit 4
        0b01101101, // Digit 5
        0b01111101, // Digit 6
        0b00000111, // Digit 7
        0b01111111, // Digit 8
        0b01101111, // Digit 9
    ];
    const SPACE: u8 = 0b00000000;

    const ASCII_TABLE: [u8; 128] = [
        // Control characters (0-31) + space (32)
        0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, // 0-4
        0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, // 5-9
        0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, // 10-14
        0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, // 15-19
        0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, //  20-24
        0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, //  25-29
        0b00000000, 0b00000000, 0b00000000, // 30-32
        // Symbols (33-47)
        0b10000110, // !
        0b00000000, // "
        0b00000000, // #
        0b00000000, // $
        0b00000000, // %
        0b00000000, // &
        0b00000000, // '
        0b00000000, // (
        0b00000000, // )
        0b00000000, // *
        0b00000000, // +
        0b00000000, // ,
        0b01000000, // -
        0b10000000, // .
        0b00000000, // /
        // Numbers (48-57)
        0b00111111, // 0
        0b00000110, // 1
        0b01011011, // 2
        0b01001111, // 3
        0b01100110, // 4
        0b01101101, // 5
        0b01111101, // 6
        0b00000111, // 7
        0b01111111, // 8
        0b01101111, // 9
        // Symbols (58-64)
        0b00000000, // :
        0b00000000, // ;
        0b00000000, // <
        0b00000000, // =
        0b00000000, // >
        0b00000000, // ?
        0b00000000, // @
        // Uppercase letters (65-90)
        0b01110111, // A
        0b01111100, // B (same as b)
        0b00111001, // C
        0b01011110, // D (same as d)
        0b01111001, // E
        0b01110001, // F
        0b00111101, // G (same as 9)
        0b01110110, // H
        0b00000110, // I (same as 1)
        0b00011110, // J
        0b01110110, // K (approximation)
        0b00111000, // L
        0b00010101, // M (arbitrary, no good match)
        0b01010100, // N
        0b00111111, // O (same as 0)
        0b01110011, // P
        0b01100111, // Q
        0b01010000, // R
        0b01101101, // S (same as 5)
        0b01111000, // T
        0b00111110, // U
        0b00101010, // V (arbitrary, no good match)
        0b00011101, // W (arbitrary, no good match)
        0b01110110, // X (same as H)
        0b01101110, // Y
        0b01011011, // Z (same as 2)
        // Symbols (91-96)
        0b00111001, // [
        0b00000000, // \
        0b00001111, // ]
        0b00000000, // ^
        0b00001000, // _
        0b00000000, // `
        // Lowercase letters (97-122), reusing uppercase for simplicity
        0b01110111, // A
        0b01111100, // B (same as b)
        0b00111001, // C
        0b01011110, // D (same as d)
        0b01111001, // E
        0b01110001, // F
        0b00111101, // G (same as 9)
        0b01110100, // H
        0b00000110, // I (same as 1)
        0b00011110, // J
        0b01110110, // K (approximation)
        0b00111000, // L
        0b00010101, // M (arbitrary, no good match)
        0b01010100, // N
        0b00111111, // O (same as 0)
        0b01110011, // P
        0b01100111, // Q
        0b01010000, // R
        0b01101101, // S (same as 5)
        0b01111000, // T
        0b00111110, // U
        0b00101010, // V (arbitrary, no good match)
        0b00011101, // W (arbitrary, no good match)
        0b01110110, // X (same as H)
        0b01101110, // Y
        0b01011011, // Z (same as 2)
        // Placeholder for simplicity
        0b00111001, // '{' (123)
        0b00000110, // '|' (124)
        0b00001111, // '}' (125)
        0b01000000, // '~' (126)
        0b00000000, // delete (127)
    ];
}

// cmk try to make generic?
// cmk linear could be a method on RangeMapBlaze
pub fn linear(
    range_map_blaze: &RangeMapBlaze<i32, u8>,
    scale: i32,
    shift: i32,
) -> RangeMapBlaze<i32, u8> {
    if range_map_blaze.is_empty() {
        return RangeMapBlaze::new();
    }

    let first = range_map_blaze.first_key_value().unwrap().0;
    let last = range_map_blaze.last_key_value().unwrap().0;

    range_map_blaze
        .range_values()
        .map(|range_value| {
            let (start, end) = range_value.range.clone().into_inner();
            let mut a = (start - first) * scale.abs() + first;
            let mut b = (end + 1 - first) * scale.abs() + first - 1;
            let last = (last + 1 - first) * scale.abs() + first - 1;
            if scale < 0 {
                (a, b) = (last - b + first, last - a + first);
            }
            let new_range = a + shift..=b + shift;
            (new_range, range_value.value)
        })
        .collect()
}

pub fn linear_wide<const DIGIT_COUNT: usize>(
    range_map_blaze: &RangeMapBlaze<i32, [u8; DIGIT_COUNT]>,
    scale: i32,
    shift: i32,
) -> RangeMapBlaze<i32, [u8; DIGIT_COUNT]> {
    if range_map_blaze.is_empty() {
        return RangeMapBlaze::new();
    }

    let first = range_map_blaze.first_key_value().unwrap().0;
    let last = range_map_blaze.last_key_value().unwrap().0;

    range_map_blaze
        .range_values()
        .map(|range_value| {
            let (start, end) = range_value.range.clone().into_inner();
            let mut a = (start - first) * scale.abs() + first;
            let mut b = (end + 1 - first) * scale.abs() + first - 1;
            let last = (last + 1 - first) * scale.abs() + first - 1;
            if scale < 0 {
                (a, b) = (last - b + first, last - a + first);
            }
            let new_range = a + shift..=b + shift;
            (new_range, range_value.value)
        })
        .collect()
}
