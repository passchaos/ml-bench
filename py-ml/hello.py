from jinja2.nodes import Tuple
import torch
import time

device = torch.device("cuda:0")
dtype = torch.bfloat16

m = 4096 * 4
n = 4096
k = 4096

def create_rand_tensors():
    a = torch.rand(m, k, dtype=dtype, device=device)
    b = torch.rand(k, n, dtype=dtype, device= device)
    c = torch.rand(m, n, dtype=dtype, device=device)

    return a, b, c

@torch.no_grad()
def bench_logic():
    a, b, c = create_rand_tensors()

    # torch.cuda.synchronize()
    # print(f"a: {a.dtype}")
    res = a.matmul(b)
    torch.cuda.synchronize()

    return res


# f32 =============================================
# no compile:
# highest: 3.27ms 3.05ms(only compute)
# high: 2.53ms
# medium: 2.53ms 2.35ms(only compute)
#
# compile:
# highest: 3.33ms
# high: 2.10ms
# medium: 2.10ms
#

# f16 =============================================
# no compile:
# highest: 3.27ms 3.05ms(only compute)
# high: 2.53ms
# medium: 2.53ms 2.35ms(only compute)
#
# compile:
# highest: 3.33ms
# high: 2.10ms
# medium: 2.10ms
#

# bfloat16 =============================================
# no compile: 1.01ms (only compute)
# compile: 1.08ms (only compute)

def main_logic():
    bl = torch.compile(bench_logic, mode='max-autotune')
    # bl = bench_logic

    # warmup
    res = bl()

    count = 1000

    costs = []

    for _ in range(count):
        a, b, c = create_rand_tensors()
        torch.cuda.synchronize()

        begin = time.time()

        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        # ) as prof:
        res = a.matmul(b) + c        # res = bl()
        torch.cuda.synchronize()

        elapsed = (time.time() - begin) * 1000

        # if res is None:
        #     continue

        print(f"torch elapsed: {elapsed}ms res shape: {res[100][-1]}")
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        costs.append(elapsed)

    try:
        costs.remove(max(costs))
        costs.remove(min(costs))

        mean = sum(costs) / len(costs)
        print(f"Mean elapsed time: {mean}ms")
    except ValueError:
        pass


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(device)

    torch.set_float32_matmul_precision('medium')
    with torch.no_grad():
        main_logic()
