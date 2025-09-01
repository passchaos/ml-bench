Refernce for Rust ml framework

# 2025-09-01
## matmul
add explicit cuda synchronize in bench loop, candle and burn-cubecl differences
between the results before and after is not significant, but the time consumption
of PyTorch has suddenly increased to 2.68ms
|framework|4090d|
|---|---|
|torch+compile|2.68ms|
|candle|16.4ms|
|burn-cubecl|1.46ms|

# 2025-08-29
## matmul
use bf16 floating point number
|framework|4090d|
|---|---|
|torch+compile|0.048ms|
|candle|16.3ms|
|burn-cubecl|1.51ms|

# 2025-08-28
## matmul
use f32 floating point number
|framework|4090d|
|---|---|
|torch+compile|0.048ms|
|candle|18.9ms|
|burn-cubecl|2.95ms|
