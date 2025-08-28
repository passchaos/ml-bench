用于测试不同机器学习实现的性能，尤其是不同语言的，为优Rust的ML生态做参考

# 2025-08-28
## 矩阵乘算法
|framework|3090|4090d|
|---|---|---|
|torch+compile|121ms|43ms|
|candle|207ms|127ms|
|burn-cubecl|0ms|0.118ms|
