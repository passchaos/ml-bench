use std::time::Instant;

#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "burn")]
mod burn {
    use burn_tensor::bf16;
    use burn_tensor::{Distribution, Tensor, backend::Backend};

    #[cfg(feature = "burn-tch")]
    mod tch_inner {
        pub use burn_tch::{LibTorch, LibTorchDevice};

        pub(super) type Tch<F = f32> = LibTorch<F>;

        pub fn device() -> LibTorchDevice {
            LibTorchDevice::Cuda(0)
        }
    }

    #[cfg(feature = "burn-candle")]
    mod candle_inner {
        pub use burn_candle::{Candle as CandleBackend, CandleDevice};

        pub(super) type Candle<F = super::bf16, I = u32> = CandleBackend<F, I>;

        pub fn device() -> CandleDevice {
            CandleDevice::cuda(0)
        }
    }

    #[cfg(feature = "burn-cubecl")]
    mod cubecl_inner {
        use burn_cubecl::CubeBackend;
        use burn_cubecl::cubecl::{cuda::CudaRuntime, wgpu::WgpuRuntime};

        pub(super) type Cuda<F = super::bf16, I = i32> = CubeBackend<CudaRuntime, F, I, u8>;
        pub(super) type Wgpu<F = super::bf16, I = i32> = CubeBackend<WgpuRuntime, F, I, u8>;

        pub fn device() -> burn_cubecl::cubecl::cuda::CudaDevice {
            burn_cubecl::cubecl::cuda::CudaDevice::new(0)
        }
    }

    #[cfg(feature = "burn-tch")]
    type B = tch_inner::Tch;
    #[cfg(feature = "burn-cubecl")]
    // type B = burn_fusion::Fusion<cubecl_inner::Cuda>;
    type B = cubecl_inner::Cuda;
    #[cfg(feature = "burn-candle")]
    type B = candle_inner::Candle;

    #[cfg(feature = "burn-tch")]
    pub type Device = tch_inner::LibTorchDevice;

    #[cfg(feature = "burn-cubecl")]
    pub type Device = burn_cubecl::cubecl::cuda::CudaDevice;

    #[cfg(feature = "burn-candle")]
    pub type Device = candle_inner::CandleDevice;

    pub fn device() -> Device {
        #[cfg(feature = "burn-tch")]
        {
            tch_inner::device()
        }

        #[cfg(feature = "burn-cubecl")]
        {
            cubecl_inner::device()
        }

        #[cfg(feature = "burn-candle")]
        {
            candle_inner::device()
        }
    }

    pub fn sync(device: &Device) {
        B::sync(device);
    }

    pub fn matmul_cuda(device: &Device) -> Tensor<B, 2> {
        let tensor1 = Tensor::<B, 2>::random([60000, 784], Distribution::Default, device);
        let tensor2 = Tensor::<B, 2>::random([784, 1000], Distribution::Default, device);

        // println!("tensor1: {tensor1}");
        let tensor3 = Tensor::<B, 2>::random([1, 1000], Distribution::Default, device);

        B::sync(device);

        let res = tensor1.matmul(tensor2) + tensor3;

        B::sync(device);
        res
    }
}

#[cfg(feature = "candle")]
mod candle {
    pub use candle_core::Device;
    use candle_core::FloatDType;
    use candle_core::Tensor;
    use half::bf16;
    // use candle_core::bf16;
    pub fn sync(device: &Device) {
        device.synchronize().unwrap();
    }

    pub fn device() -> Device {
        Device::new_cuda(0).unwrap()
    }

    pub fn matmul_cuda(device: &Device) -> Tensor {
        let zero = bf16::ZERO;
        let one = bf16::ONE;

        let tensor1 = Tensor::rand::<_, bf16>(zero, one, &[60000, 784], device).unwrap();
        let tensor2 = Tensor::rand::<_, bf16>(zero, one, &[784, 1000], device).unwrap();
        // println!("tensor1: {tensor1}");
        let tensor3 = Tensor::rand::<_, bf16>(zero, one, &[1, 1000], device).unwrap();

        device.synchronize().unwrap();

        // println!("tensor1: {tensor1}");

        let res = tensor1.matmul(&tensor2).unwrap().broadcast_add(&tensor3);

        device.synchronize().unwrap();

        res.unwrap()
    }
}

#[cfg(feature = "burn")]
type Device = burn::Device;

#[cfg(feature = "candle")]
type Device = candle::Device;

#[cfg(feature = "burn")]
fn device() -> Device {
    burn::device()
}

#[cfg(feature = "candle")]
fn device() -> Device {
    candle::device()
}

fn sync(device: &Device) {
    #[cfg(feature = "candle")]
    {
        candle::sync(device);
    }

    #[cfg(feature = "burn")]
    {
        burn::sync(device);
    }
}

fn matmul(device: &Device) {
    #[cfg(feature = "candle")]
    {
        let res = candle::matmul_cuda(device);
        // println!("res: {res}");
    }

    #[cfg(feature = "burn")]
    {
        use burn_tensor::s;
        let res = burn::matmul_cuda(device);
        println!("res: {:?}", res.slice(s![100, -1]).into_scalar());
    }
}

fn main() {
    let device = device();
    // warmup
    matmul(&device);

    let count = 10;
    let mut costs = vec![];

    for _ in 0..count {
        let begin = Instant::now();

        matmul(&device);
        sync(&device);
        let elapsed = begin.elapsed().as_secs_f32() * 1000.0;
        println!("elapsed: {elapsed}ms");

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
