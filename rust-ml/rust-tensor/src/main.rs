use std::{mem::MaybeUninit, time::Instant};

#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

const M: usize = 128 * 32;
const K: usize = 128 * 32;
const N: usize = 128 * 32;

use burn_tensor::{Distribution, Tensor, backend::Backend};

use burn_cubecl::CubeBackend;
use cudarc::{
    cublas::{
        CudaBlas, Gemm, GemmConfig,
        result::gemm_ex,
        sys::{self, cublasCreate_v2, cublasGemmEx, cublasSetMathMode},
    },
    driver::{CudaContext, DevicePtr, DevicePtrMut},
    runtime::sys::cudaStreamSynchronize,
};

type CudaBackend = CubeBackend<burn_cubecl::cubecl::cuda::CudaRuntime, burn_tensor::bf16, i32, u8>;
type Device = burn_cubecl::cubecl::cuda::CudaDevice;

pub fn device() -> Device {
    Device::new(0)
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
fn run_cublas() -> Vec<f32> {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).unwrap();

    let m = 1024 * 4;
    let n = 1024 * 4;
    let k = 1024 * 4;

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

    // let a_d = stream.memcpy_stod(&a).unwrap();
    // let b_d = stream.memcpy_stod(&b).unwrap();
    // // let c_d = stream.memcpy_stod(&c).unwrap();
    // let mut c_d = stream.alloc_zeros::<half::bf16>(m * n).unwrap();

    // let (a, _record_a) = a_d.device_ptr(&stream);
    // let (b, _record_b) = b_d.device_ptr(&stream);
    // let (c, _record_c) = c_d.device_ptr_mut(&stream);

    // let begin = Instant::now();

    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cublasCreate_v2(handle.as_mut_ptr()).result().unwrap();
    }
    let handle = unsafe { handle.assume_init() };

    unsafe {
        cublasSetMathMode(handle, sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH);
    }
    // let res = unsafe {
    //     cublasGemmEx(
    //         handle.assume_init(),
    //         cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
    //         cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
    //         m as i32,
    //         n as i32,
    //         k as i32,
    //         &(1.0f32) as *const f32 as *const _,
    //         a as *const _,
    //         cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
    //         m as i32,
    //         b as *const _,
    //         cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
    //         n as i32,
    //         &(1.0f32) as *const f32 as *const _,
    //         c as *mut _,
    //         cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
    //         n as i32,
    //         cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
    //         cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
    //     )
    // };
    // // .unwrap();
    // // stream.memcpy

    // stream.synchronize().unwrap();

    let mut costs = Vec::new();
    for _ in 0..10 {
        let a_d = stream.memcpy_stod(&a).unwrap();
        let b_d = stream.memcpy_stod(&b).unwrap();
        // let c_d = stream.memcpy_stod(&c).unwrap();
        let mut c_d = stream.alloc_zeros::<half::bf16>(m * n).unwrap();

        let begin = Instant::now();

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
                n as i32,
                &(1.0f32) as *const f32 as *const _,
                c as *mut _,
                cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        // let res = unsafe {
        //     blas.gemm(
        //         GemmConfig {
        //             transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        //             transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        //             m: m as i32,
        //             n: n as i32,
        //             k: k as i32,
        //             alpha: half::bf16::from_f32(1.0),
        //             lda: n as i32,
        //             ldb: k as i32,
        //             beta: half::bf16::from_f32(0.0),
        //             ldc: n as i32,
        //         },
        //         &a_d,
        //         &b_d,
        //         &mut c_d,
        //     )
        // }
        // .unwrap();
        // stream.memcpy

        unsafe {
            cudarc::driver::sys::cuStreamSynchronize(stream.cu_stream());
        }
        // cudaStreamSynchronize(stream.cu_stream());
        // stream.synchronize().unwrap();

        let cost = begin.elapsed().as_secs_f32() * 1000.0;
        costs.push(cost);
    }
    costs
}

fn run_cubecl() -> Vec<f32> {
    let device = device();
    // warmup

    matmul_cuda_with_init::<CudaBackend>(&device);
    CudaBackend::sync(&device);

    let count = 100;
    let mut costs = vec![];

    let (a, b, c) = rand_tensor::<CudaBackend>(&device);
    CudaBackend::sync(&device);

    for _ in 0..count {
        let a = a.clone();
        let b = b.clone();
        let c = c.clone();
        CudaBackend::sync(&device);

        let begin = Instant::now();

        let res = a.matmul(b) + c;
        CudaBackend::sync(&device);

        let elapsed = begin.elapsed().as_secs_f32() * 1000.0;
        println!("elapsed: {elapsed}ms");

        costs.push(elapsed);
    }
    costs
}

fn main() {
    // let mut costs = run_cubecl();
    let mut costs = run_cublas();
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
