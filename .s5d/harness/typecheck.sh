#!/bin/sh
# s5d typecheck gate: cargo check for the supermut Rust core.
export PATH="/Users/random1st/.cargo/bin:/usr/bin:/bin"
export HOME="/Users/random1st"
exec cargo check
