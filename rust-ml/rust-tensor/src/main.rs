use std::{mem::MaybeUninit, sync::Arc, time::Instant};

use clap::{Parser, ValueEnum};

#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

const M: usize = 128 * 16 * 4 * 2;
const K: usize = 128 * 32;
const N: usize = 128 * 32;

use burn_tensor::{Distribution, Tensor, backend::Backend};

use burn_cubecl::CubeBackend;

#[cfg(not(target_os = "macos"))]
use cudarc::{
    cublas::{
        CudaBlas, Gemm, GemmConfig,
        result::gemm_ex,
        sys::{self, cublasCreate_v2, cublasGemmEx, cublasHandle_t, cublasSetMathMode},
    },
    driver::{
        CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
        sys::{cuEventElapsedTime, cuEventRecord},
    },
    runtime::sys::{
        cudaDeviceSynchronize, cudaEventCreate, cudaEventRecord, cudaFree, cudaStreamSynchronize,
    },
};
use half::bf16;

#[cfg(not(target_os = "macos"))]
type Back = CubeBackend<burn_cubecl::cubecl::cuda::CudaRuntime, burn_tensor::bf16, i32, u8>;
#[cfg(not(target_os = "macos"))]
type Device = burn_cubecl::cubecl::cuda::CudaDevice;

#[cfg(target_os = "macos")]
type Back = CubeBackend<burn_cubecl::cubecl::wgpu::WgpuRuntime, f32, i32, u8>;

#[cfg(target_os = "macos")]
type Device = burn_cubecl::cubecl::wgpu::WgpuDevice;

pub fn device() -> Device {
    #[cfg(target_os = "macos")]
    {
        Device::default()
    }

    #[cfg(not(target_os = "macos"))]
    {
        Device::new(0)
    }
}

pub fn matmul_cuda<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 2>, c: Tensor<B, 2>) -> Tensor<B, 2> {
    a.matmul(b)
}

pub fn rand_tensor<B: Backend>(device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let tensor1 = Tensor::<B, 2>::random([M, K], Distribution::Default, device);
    let tensor2 = Tensor::<B, 2>::random([K, N], Distribution::Default, device);

    let tensor3 = Tensor::<B, 2>::random([M, N], Distribution::Default, device);

    (tensor1, tensor2, tensor3)
}

pub fn matmul_cuda_with_init<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
    let (tensor1, tensor2, tensor3) = rand_tensor(device);
    let res = matmul_cuda(tensor1, tensor2, tensor3);

    res
}

// cubecl==========================================
// f32: 3.19ms (only compute)
// bf16: 1.57ms (only compute)
// cublas==========================================
// bf16: 1.08ms
#[cfg(not(target_os = "macos"))]
fn run_cublas(use_raw: bool) -> Vec<f32> {
    let ctx = CudaContext::new(0).unwrap();

    let m = M;
    let n = N;
    let k = K;

    let a: Vec<_> = rand::random_iter()
        .take(m * k)
        .map(half::bf16::from_f32)
        .collect();

    let b: Vec<_> = rand::random_iter()
        .take(k * n)
        .map(half::bf16::from_f32)
        .collect();

    let c: Vec<_> = rand::random_iter()
        .take(m * n)
        .map(half::bf16::from_f32)
        .collect();

    let stream = ctx.default_stream();

    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cublasCreate_v2(handle.as_mut_ptr()).result().unwrap();
    }
    let handle = unsafe { handle.assume_init() };

    unsafe {
        cublasSetMathMode(handle, sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH);
    }

    let blas = CudaBlas::new(stream.clone()).unwrap();

    let (a_d, b_d, mut c_d) = {
        let a_d = stream.memcpy_stod(&a).unwrap();
        let b_d = stream.memcpy_stod(&b).unwrap();
        // let c_d = stream.memcpy_stod(&c).unwrap();
        let c_d = stream.memcpy_stod(&c).unwrap();
        (a_d, b_d, c_d)
    };

    let event_flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;

    let mut costs = Vec::new();
    for _ in 0..1000 {
        let (a_d, b_d, mut c_d) = {
            let a_d = stream.memcpy_stod(&a).unwrap();
            let b_d = stream.memcpy_stod(&b).unwrap();
            // let c_d = stream.memcpy_stod(&c).unwrap();
            let c_d = stream.memcpy_stod(&c).unwrap();
            (a_d, b_d, c_d)
        };

        unsafe { cudaDeviceSynchronize() };

        let begin_event = ctx.new_event(Some(event_flags)).unwrap();
        let end_event = ctx.new_event(Some(event_flags)).unwrap();

        begin_event.record(&stream).unwrap();

        let begin = Instant::now();
        if use_raw {
            let (a, _record_a) = a_d.device_ptr(&stream);
            let (b, _record_b) = b_d.device_ptr(&stream);
            let (c, _record_c) = c_d.device_ptr_mut(&stream);

            let res = unsafe {
                cublasGemmEx(
                    handle,
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    m as i32,
                    n as i32,
                    k as i32,
                    &(1.0f32) as *const f32 as *const _,
                    a as *const _,
                    cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                    m as i32,
                    b as *const _,
                    cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                    k as i32,
                    &(1.0f32) as *const f32 as *const _,
                    c as *mut _,
                    cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                    m as i32,
                    cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            };

            end_event.record(&stream).unwrap();
            unsafe { cudaDeviceSynchronize() };
        } else {
            unsafe {
                blas.gemm(
                    GemmConfig {
                        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                        m: m as i32,
                        n: n as i32,
                        k: k as i32,
                        alpha: half::bf16::from_f32(1.0),
                        lda: m as i32,
                        ldb: k as i32,
                        beta: half::bf16::from_f32(1.0),
                        ldc: m as i32,
                    },
                    &a_d,
                    &b_d,
                    &mut c_d,
                )
            }
            .unwrap();

            end_event.record(&stream).unwrap();
            stream.synchronize().unwrap();
        }

        let elapsed = begin.elapsed().as_secs_f32() * 1000.0;

        let cost = begin_event.elapsed_ms(&end_event).unwrap();
        costs.push(cost);
        println!("Time for compute: {cost}ms {elapsed}ms");
    }

    costs
}

fn run_candle() -> Vec<f32> {
    use candle_core::Tensor;

    #[cfg(not(target_os = "macos"))]
    type Dtype = bf16;
    #[cfg(target_os = "macos")]
    type Dtype = f32;

    let zero = Dtype::from_f32(0.0);
    let one = Dtype::from_f32(1.0);

    #[cfg(not(target_os = "macos"))]
    let device = candle_core::Device::new_cuda(0).unwrap();

    #[cfg(target_os = "macos")]
    let device = candle_core::Device::new_metal(0).unwrap();

    let tensor1 = Tensor::rand::<_, Dtype>(zero, one, &[M, K], &device).unwrap();
    let tensor2 = Tensor::rand::<_, Dtype>(zero, one, &[K, N], &device).unwrap();

    let tensor3 = Tensor::rand::<_, Dtype>(zero, one, &[M, N], &device).unwrap();

    device.synchronize().unwrap();

    let mut costs = Vec::new();
    for _ in 0..1000 {
        device.synchronize().unwrap();

        let begin = Instant::now();
        let b = tensor1.matmul(&tensor2).unwrap() + &tensor3;

        device.synchronize().unwrap();
        let elapsed = begin.elapsed().as_secs_f32() * 1000.0;
        println!("Time for compute: {elapsed}ms");
        costs.push(elapsed);
    }

    costs
}

fn run_cubecl() -> Vec<f32> {
    let device = device();
    // warmup

    matmul_cuda_with_init::<Back>(&device);
    Back::sync(&device);

    let count = 1000;
    let mut costs = vec![];

    let (a, b, c) = rand_tensor::<Back>(&device);
    Back::sync(&device);

    for _ in 0..count {
        let a = a.clone();
        let b = b.clone();
        let c = c.clone();
        Back::sync(&device);

        let begin = Instant::now();

        let res = a.matmul(b) + c;
        Back::sync(&device);

        let elapsed = begin.elapsed().as_secs_f32() * 1000.0;
        println!("elapsed: {elapsed}ms");

        costs.push(elapsed);
    }
    costs
}

#[derive(Debug, Clone, ValueEnum)]
enum ComputeKind {
    CublasSafe,
    CublasRaw,
    Candle,
    Cubecl,
}

#[derive(Parser, Debug)]
struct Args {
    kind: ComputeKind,
}

fn main() {
    let cli = Args::parse();

    let mut costs = match cli.kind {
        ComputeKind::CublasRaw => {
            #[cfg(not(target_os = "macos"))]
            {
                run_cublas(true)
            }

            #[cfg(target_os = "macos")]
            vec![]
        }

        ComputeKind::CublasSafe => {
            #[cfg(not(target_os = "macos"))]
            {
                run_cublas(false)
            }

            #[cfg(target_os = "macos")]
            vec![]
        }
        ComputeKind::Candle => run_candle(),
        // _ => run_candle(),
        ComputeKind::Cubecl => run_cubecl(),
    };

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
