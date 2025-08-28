import torch
import time

device = torch.device("cuda:0")

@torch.no_grad()
def bench_logic():
    a = torch.rand(60000, 784).to(device)
    b = torch.rand(784, 1000).to(device)
    c = torch.rand(1, 1000).to(device)

    torch.cuda.synchronize()
    res = a.matmul(b) + c
    torch.cuda.synchronize()

    return res

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(device)
    torch.set_float32_matmul_precision('high')
    bl = torch.compile(bench_logic, mode='max-autotune')

    # warmup
    res = bl()

    count = 10

    sum = 0.0

    for _ in range(count):
        begin = time.time()
        res = bl()
        elapsed = (time.time() - begin) * 1000
        print(f"torch elapsed: {elapsed}ms")
        sum += elapsed

        torch.cuda.empty_cache()

    mean = sum / count
    print(f"Mean elapsed time: {mean}ms")
