use std::time::Instant;

#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "burn")]
mod burn {
    use burn_cubecl::CubeBackend;
    use burn_tensor::{Distribution, Tensor, backend::Backend};
    use cubecl::cuda::CudaRuntime;
    use cubecl::wgpu::WgpuRuntime;
    type Cuda<F = f32, I = i32> = CubeBackend<CudaRuntime, F, I, u8>;
    type CudaFusion<F = f32, I = i32> = burn_fusion::Fusion<CubeBackend<CudaRuntime, F, I, u8>>;
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
    use candle_core::{Device, Tensor};

    pub fn matmul_cuda() -> Tensor {
        let device = Device::new_cuda(0).unwrap();

        let tensor1 = Tensor::rand(0.0, 1.0, &[60000, 784], &device).unwrap();
        let tensor2 = Tensor::rand(0.0, 1.0, &[784, 1000], &device).unwrap();
        // println!("tensor1: {tensor1}");
        let tensor3 = Tensor::rand(0.0, 1.0, &[1, 1000], &device).unwrap();

        device.synchronize().unwrap();
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
    let mut sum = 0;

    for _ in 0..count {
        let begin = Instant::now();

        matmul();
        let elapsed = begin.elapsed().as_millis();

        sum += elapsed;
        println!("burn elapsed: {elapsed}ms");
    }

    let mean = (sum as f64) / (count as f64);
    println!("Mean elapsed time: {mean}ms");
}
