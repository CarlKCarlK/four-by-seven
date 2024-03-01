# Useful Notes

* <https://github.com/U007D/blinky>
* <https://embassy.dev/book/dev/getting_started.html>
* <https://www.hannobraun.com/getting-started/>

```cmd
github clone https://github.com/U007D/blinky
cd blinky\blinky
rustup override set stable
rustup target add thumbv6m-none-eabi
cargo install elf2uf2-rs svd2rust

```

 > And remember to hold down the Pico's button while powering it on if you want to be able to reprogram it.  cargo run (as redefined in .cargo/config.toml) will:

* cross-compile your code,
* convert the executable to RPi format,
* copy the converted executable to your Pi,
* flash your board and
* run your program for you in 1 step.
