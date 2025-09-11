Refernce for Rust ml framework

# 2025-09-09
## matmul
use matrix shape opt for cache line

A*B + C

A.shape: 235\*256 x 4\*256

B.shape: 4\*256 x 5\*256

C.shape: 1 x 5 \* 256


|framework|4090d|
|---|---|
|torch|1.78ms|
|torch+compile|1.37ms|
|burn-tch|2.10ms|
|burn-cubecl|2.30ms|

# 2025-09-01
## matmul
reuse device for rust ml; use non-compile mode for pytorch

|framework|4090d|
|---|---|
|torch|1.22ms|
|torch+compile|1.01ms|
|candle|11.9ms|
|burn-tch|3.52ms|
|burn-cubecl|1.56ms|
## cuda trace
cubecl don't use the new hardware gemm instruction
### candle
![](https://raw.githubusercontent.com/passchaos/sundry/main/images/20250908123006523.png)
### cubecl
![](https://raw.githubusercontent.com/passchaos/sundry/main/images/20250908123220416.png)


# 2025-09-01
## matmul
add explicit cuda synchronize in bench loop, candle and burn-cubecl differences
between the results before and after is not significant, but the time consumption
of PyTorch has suddenly increased to 2.68ms
|framework|4090d|
|---|---|
|torch+compile|0.90ms|
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
