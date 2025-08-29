use std::time::Instant;

#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "burn")]
mod burn {
    use burn_cubecl::CubeBackend;
    use burn_tensor::{Distribution, Tensor, backend::Backend};
    use cubecl::cuda::CudaRuntime;
    use cubecl::wgpu::WgpuRuntime;
    use half::bf16;

    type Cuda<F = bf16, I = i32> = CubeBackend<CudaRuntime, F, I, u8>;
    type CudaFusion<F = bf16, I = i32> = burn_fusion::Fusion<CubeBackend<CudaRuntime, F, I, u8>>;
    type Wgpu<F = f32, I = i32> = CubeBackend<WgpuRuntime, F, I, u8>;

    type B = CudaFusion;

    pub fn matmul_cuda() -> Tensor<B, 2> {
        let device = Default::default();
        let tensor1 = Tensor::<B, 2>::random([60000, 784], Distribution::Default, &device);
        let tensor2 = Tensor::<B, 2>::random([784, 1000], Distribution::Default, &device);

        // println!("tensor1: {tensor1}");
        let tensor3 = Tensor::<B, 2>::random([1, 1000], Distribution::Default, &device);

        B::sync(&device);

        let res = tensor1.matmul(tensor2) + tensor3;

        B::sync(&device);
        res
    }
}

#[cfg(feature = "candle")]
mod candle {
    use candle_core::FloatDType;
    use candle_core::{Device, Tensor};
    use half::bf16;
    // use candle_core::bf16;

    pub fn matmul_cuda() -> Tensor {
        let device = Device::new_cuda(0).unwrap();

        let zero = bf16::ZERO;
        let one = bf16::ONE;

        let tensor1 = Tensor::rand::<_, bf16>(zero, one, &[60000, 784], &device).unwrap();
        let tensor2 = Tensor::rand::<_, bf16>(zero, one, &[784, 1000], &device).unwrap();
        // println!("tensor1: {tensor1}");
        let tensor3 = Tensor::rand::<_, bf16>(zero, one, &[1, 1000], &device).unwrap();

        device.synchronize().unwrap();

        // println!("tensor1: {tensor1}");

        let res = tensor1.matmul(&tensor2).unwrap().broadcast_add(&tensor3);

        device.synchronize().unwrap();

        res.unwrap()
    }
}

fn matmul() {
    #[cfg(feature = "candle")]
    {
        let res = candle::matmul_cuda();
        // println!("res: {res}");
    }

    #[cfg(feature = "burn")]
    {
        let res = burn::matmul_cuda();
        // println!("res: {:?}", res.shape());
    }
}

fn main() {
    // warmup
    matmul();

    let count = 10;
    let mut costs = vec![];

    for _ in 0..count {
        let begin = Instant::now();

        matmul();
        let elapsed = begin.elapsed().as_secs_f32() * 1000.0;

        costs.push(elapsed);
    }

    let (max_idx, _) = costs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();

    costs.remove(max_idx);

    let (min_idx, _) = costs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();

    costs.remove(min_idx);

    let len = costs.len();
    let mean = costs.iter().sum::<f32>() / (len as f32);
    println!("Mean elapsed time: {mean}ms");
}
