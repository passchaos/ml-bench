import torch
import time

device = torch.device("cuda:0")

@torch.no_grad()
def bench_logic():
    a = torch.rand(60000, 784).to(device)
    b = torch.rand(784, 1000).to(device)
    c = torch.rand(1, 1000).to(device)
    res = a.matmul(b) + c
    return res

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    bl = torch.compile(bench_logic)

    # warmup
    res = bl()

    for _ in range(10):
        begin = time.time()
        res = bl()
        elapsed = time.time() - begin
        print(f"torch elapsed: {elapsed * 1000:.2f}ms")
        torch.cuda.empty_cache()
