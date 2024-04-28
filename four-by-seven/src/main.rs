#![no_std]
#![no_main]

extern crate alloc;

#[global_allocator]
static ALLOCATOR: CortexMHeap = CortexMHeap::empty();
const HEAP_SIZE: usize = 1024 * 64; // in bytes

use core::{
    array,
    ops::{Index, IndexMut},
};

use alloc_cortex_m::CortexMHeap;
use defmt::unwrap;
use embassy_executor::Spawner;
use embassy_futures::select::{select, Either};
use embassy_rp::gpio;
use embassy_sync::blocking_mutex::raw::ThreadModeRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_time::{Duration, Timer};
use embedded_hal_1::digital::OutputPin;
use gpio::Level;
use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};
use static_cell::StaticCell;
use {defmt_rtt as _, panic_probe as _}; // Adjust the import path according to your setup

pub const DIGIT_COUNT: usize = 4;

pub struct VirtualDisplay {
    mutex_digits: Mutex<ThreadModeRawMutex, Option<DigitArray>>,
}

static VIRTUAL_DISPLAY: VirtualDisplay = VirtualDisplay {
    mutex_digits: Mutex::new(None),
};

impl VirtualDisplay {
    async fn init(&'static self) {
        let mut mutex_digits = VIRTUAL_DISPLAY.mutex_digits.lock().await;
        *mutex_digits = Some(DigitArray::default());
    }

    async fn bool_iter(&'static self, digit_index: usize) -> array::IntoIter<bool, 8> {
        // inner scope to release the lock
        // cmk other versions of MUTEX use a closure
        let digit_array = self.mutex_digits.lock().await;
        let digit_array = digit_array.as_ref().unwrap();
        digit_array.bool_iter(digit_index)
    }

    async fn set_all_digits(&'static self, all: [u8; DIGIT_COUNT]) {
        let mut digit_array = VIRTUAL_DISPLAY.mutex_digits.lock().await;
        let digit_array = digit_array.as_mut().unwrap();
        digit_array.set_all_digits(&all);
    }
}

pub struct DigitArray {
    digits: [u8; DIGIT_COUNT],
}

// default is 255 x DIGIT_COUNT
impl Default for DigitArray {
    fn default() -> Self {
        DigitArray {
            digits: [255; DIGIT_COUNT],
        }
    }
}

impl Index<usize> for DigitArray {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        &self.digits[index]
    }
}

impl IndexMut<usize> for DigitArray {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.digits[index]
    }
}

impl DigitArray {
    pub fn set_segment(&mut self, digit: usize, segment: usize, state: bool) {
        assert!(segment < 8, "segment out of range");
        if state {
            self[digit] |= 1 << segment;
        } else {
            self[digit] &= !(1 << segment);
        }
    }

    pub fn segment(&self, digit: usize, segment: usize) -> bool {
        assert!(segment < 8, "segment out of range");
        (self[digit] >> segment) & 1 == 1
    }

    pub fn set_all_digits(&mut self, &all: &[u8; DIGIT_COUNT]) {
        for (digit_index, value) in all.iter().enumerate() {
            self[digit_index] = *value;
        }
    }

    pub fn bool_iter(&self, digit: usize) -> array::IntoIter<bool, 8> {
        let mut value: u8 = self[digit];
        let mut arr = [false; 8];
        for item in arr.iter_mut() {
            *item = value & 1 == 1;
            value >>= 1;
        }
        arr.into_iter()
    }
}

#[embassy_executor::task]
#[allow(clippy::needless_range_loop)]
async fn multiplex_display(
    digit_pins: &'static mut [gpio::Output<'_>; 4],
    segment_pins: &'static mut [gpio::Output<'_>; 8],
) {
    loop {
        for digit_index in 0..DIGIT_COUNT {
            VIRTUAL_DISPLAY
                .bool_iter(digit_index)
                .await
                .zip(segment_pins.iter_mut())
                .for_each(|(state, segment_pin)| {
                    segment_pin.set_state(state.into()).unwrap();
                });
            // Activate and deactivate the digit
            digit_pins[digit_index].set_low(); // Assuming common cathode setup
            Timer::after(Duration::from_millis(5)).await;
            digit_pins[digit_index].set_high();
        }
    }
}

fn init_pins() -> (
    &'static mut [gpio::Output<'static>; 4],
    &'static mut [gpio::Output<'static>; 8],
    &'static mut gpio::Input<'static>,
    &'static mut gpio::Output<'static>,
) {
    let p: embassy_rp::Peripherals = embassy_rp::init(Default::default());
    static DIGIT_PINS: StaticCell<[gpio::Output; 4]> = StaticCell::new();
    let digit_pins = DIGIT_PINS.init([
        gpio::Output::new(p.PIN_1, Level::High),
        gpio::Output::new(p.PIN_2, Level::High),
        gpio::Output::new(p.PIN_3, Level::High),
        gpio::Output::new(p.PIN_4, Level::High),
    ]);
    static SEGMENT_PINS: StaticCell<[gpio::Output; 8]> = StaticCell::new();
    let segment_pins = SEGMENT_PINS.init([
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
    let button_pin = BUTTON_PIN.init(gpio::Input::new(p.PIN_13, gpio::Pull::Down));
    static LED0_PIN: StaticCell<gpio::Output> = StaticCell::new();
    let led0_pin = LED0_PIN.init(gpio::Output::new(p.PIN_0, Level::Low));
    let result: (
        &mut [gpio::Output; 4],
        &mut [gpio::Output; 8],
        &mut gpio::Input,
        &mut gpio::Output,
    ) = (digit_pins, segment_pins, button_pin, led0_pin);
    result
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    unsafe { ALLOCATOR.init(cortex_m_rt::heap_start() as usize, HEAP_SIZE) }
    let (digit_pins, segment_pins, button_pin, led0_pin) = init_pins();

    VIRTUAL_DISPLAY.init().await;

    unwrap!(spawner.spawn(multiplex_display(digit_pins, segment_pins)));

    let movies: [RangeMapBlaze<i32, [u8; DIGIT_COUNT]>; 2] = [circles_wide(), hello_world_wide()];

    // loop through the movies, forever
    for movie in movies.iter().cycle() {
        // Loop through the frames of the current movie
        for range_values in movie.range_values() {
            // Get the next frame of the movie (and its duration)
            let (start, end) = range_values.range.into_inner();
            let frame = *range_values.value;

            // Display the frame
            VIRTUAL_DISPLAY.set_all_digits(frame).await;

            // Show the frame for the correct duration
            let frame_count = (end + 1 - start) as u64;
            let duration = Duration::from_millis(frame_count * 1000 / FPS as u64);

            // Wait for the frame to finish or the button to be pressed
            if let Either::First(()) =
                select(Timer::after(duration), button_pin.wait_for_rising_edge()).await
            {
                // timer finished
                continue;
            }

            // wait for button to be released
            led0_pin.set_high();
            Timer::after(Duration::from_millis(5)).await; // debounce button
            button_pin.wait_for_falling_edge().await;
            led0_pin.set_low();
        }
    }
}

const FPS: i32 = 24;

fn line_to_u8_array(line: &str) -> [u8; DIGIT_COUNT] {
    let mut result = [0; DIGIT_COUNT];
    for (i, c) in line.chars().enumerate() {
        // cmk could try to go out of the array
        result[i] = Leds::ASCII_TABLE[c as usize];
    }
    result
}

pub fn hello_world_wide() -> RangeMapBlaze<i32, [u8; DIGIT_COUNT]> {
    let message = "3\n 2\n  1\n\n   H\n  He\n Hel\nHell\nello\nllo\nlo W\no Wo\n Wor\nWorl\norld\nrld\nld\nd\n";
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

pub fn circles_wide() -> RangeMapBlaze<i32, [u8; DIGIT_COUNT]> {
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

pub fn linear_wide(
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
