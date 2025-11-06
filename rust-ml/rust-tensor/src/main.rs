use std::{mem::MaybeUninit, sync::Arc, time::Instant};

#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

const M: usize = 128 * 16 * 4 * 2;
const K: usize = 128 * 32;
const N: usize = 128 * 32;

use burn_tensor::{Distribution, Tensor, backend::Backend};

use burn_cubecl::CubeBackend;
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

fn run_safe_cublas(
    ctx: Arc<CudaContext>,
    blas_handle: &CudaBlas,
    m: usize,
    n: usize,
    k: usize,
    a: &[bf16],
    b: &[bf16],
) -> f32 {
    let stream = ctx.default_stream();
    let a_d = stream.memcpy_stod(a).unwrap();
    let b_d = stream.memcpy_stod(b).unwrap();
    // let c_d = stream.memcpy_stod(&c).unwrap();
    let mut c_d = stream.alloc_zeros::<half::bf16>(m * n).unwrap();

    let event_flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
    let begin_event = ctx.new_event(Some(event_flags)).unwrap();
    let end_event = ctx.new_event(Some(event_flags)).unwrap();

    let begin = Instant::now();
    let begin_st = std::time::SystemTime::now();

    begin_event.record(&stream).unwrap();

    unsafe {
        blas_handle.gemm(
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

    let cost1 = begin.elapsed().as_micros();
    let cost2 = begin_st.elapsed().unwrap().as_micros();

    let cost = begin_event.elapsed_ms(&end_event).unwrap();

    println!("safe time for compute: {}ms {} {}", cost, cost1, cost2);
    cost
}

fn run_raw_cublas_inner(
    ctx: Arc<CudaContext>,
    blas_handle: cublasHandle_t,
    m: usize,
    n: usize,
    k: usize,
    a: &[bf16],
    b: &[bf16],
) -> f32 {
    let stream = ctx.default_stream();
    let a_d = stream.memcpy_stod(a).unwrap();
    let b_d = stream.memcpy_stod(b).unwrap();
    // let c_d = stream.memcpy_stod(&c).unwrap();
    let mut c_d = stream.alloc_zeros::<half::bf16>(m * n).unwrap();

    let (a, _record_a) = a_d.device_ptr(&stream);
    let (b, _record_b) = b_d.device_ptr(&stream);
    let (c, _record_c) = c_d.device_ptr_mut(&stream);

    let mut begin_event = MaybeUninit::uninit();

    let event_flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
    unsafe { cudarc::driver::sys::cuEventCreate(begin_event.as_mut_ptr(), event_flags as u32) };
    let begin_event = unsafe { begin_event.assume_init() };

    let mut end_event = MaybeUninit::uninit();
    unsafe { cudarc::driver::sys::cuEventCreate(end_event.as_mut_ptr(), event_flags as u32) };
    let end_event = unsafe { end_event.assume_init() };

    let begin_event = ctx.new_event(Some(event_flags)).unwrap();
    let end_event = ctx.new_event(Some(event_flags)).unwrap();

    // let cost_0 = begin_event.elapsed_ms(&end_event).unwrap();

    // cudarc::runtime::sys::cudaStreamCreate(pStream)
    // cudaEventRecord(begin_event, stream.cu_stream());
    // cudaEventRecord(end_event, stream.cu_stream());

    let begin = Instant::now();

    begin_event.record(&stream).unwrap();
    // unsafe {
    //     cuEventRecord(begin_event, stream.cu_stream());
    // }

    let res = unsafe {
        cublasGemmEx(
            blas_handle,
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

    // unsafe {
    //     cuEventRecord(end_event, stream.cu_stream());
    // }

    unsafe { cudaDeviceSynchronize() };

    let cost1 = begin.elapsed().as_micros();

    // let mut cost: f32 = 0.0;
    // unsafe {
    //     cuEventElapsedTime((&mut cost) as *mut _, begin_event, end_event);
    // }

    let cost = begin_event.elapsed_ms(&end_event).unwrap();

    println!("raw time for compute: {}ms {}ms", cost, cost1);
    cost

    // unsafe {
    //     cudarc::driver::sys::cuStreamSynchronize(stream.cu_stream());
    // }
    // cudaStreamSynchronize(stream.cu_stream());
    // stream.synchronize().unwrap();
}

// cubecl==========================================
// f32: 3.19ms (only compute)
// bf16: 1.57ms (only compute)
// cublas==========================================
// bf16: 1.08ms
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

    let costs = if use_raw {
        let mut handle = MaybeUninit::uninit();
        unsafe {
            sys::cublasCreate_v2(handle.as_mut_ptr()).result().unwrap();
        }
        let handle = unsafe { handle.assume_init() };

        unsafe {
            cublasSetMathMode(handle, sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH);
        }

        let _cost = run_raw_cublas_inner(ctx.clone(), handle, m, n, k, &a, &b);

        let mut costs = Vec::new();
        for _ in 0..1000 {
            let cost = run_raw_cublas_inner(ctx.clone(), handle, m, n, k, &a, &b);
            costs.push(cost);
            println!("Time for compute: {cost}ms");
        }

        costs
    } else {
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).unwrap();

        let _cost = run_safe_cublas(ctx.clone(), &blas, m, n, k, &a, &b);

        let mut costs = Vec::new();
        for _ in 0..1000 {
            let cost = run_safe_cublas(ctx.clone(), &blas, m, n, k, &a, &b);
            costs.push(cost);
            println!("Time for compute: {cost}ms");
        }
        costs
    };

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
    let mut costs = run_cublas(false);
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
