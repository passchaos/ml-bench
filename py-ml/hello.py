import torch
import time

device = torch.device("cuda:0")

@torch.no_grad()
def bench_logic():
    a = torch.rand(60000, 784, device=device)
    b = torch.rand(784, 1000,device = device)
    c = torch.rand(1, 1000, device=device)

    torch.cuda.synchronize()
    # print(f"a: {a.dtype}")
    res = a.matmul(b) + c
    torch.cuda.synchronize()

    return res

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(device)
    torch.set_float32_matmul_precision('highest')
    bl = torch.compile(bench_logic, mode='max-autotune')

    # warmup
    res = bl()

    count = 10

    costs = []

    for _ in range(count):
        begin = time.time()
        res = bl()
        elapsed = (time.time() - begin) * 1000
        print(f"torch elapsed: {elapsed}ms")
        costs.append(elapsed)

    costs.remove(max(costs))
    costs.remove(min(costs))

    mean = sum(costs) / len(costs)
    print(f"Mean elapsed time: {mean}ms")
