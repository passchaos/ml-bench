//! SIMD/NEON example showing performance comparison

use criterion::{Criterion, criterion_group, criterion_main};
use rand::random;
use simd_learn::simple::*;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("brightness_adjustment");
    // let factor = 0.7;

    let mut pixels: Vec<_> = vec![0u8; 1920 * 1080 * 3]
        .into_iter()
        .map(|_a| random())
        .collect();

    for factor in [f32::EPSILON, 0.1, 0.5, 0.9] {
        group.bench_function(format!("normal_{factor}"), |b| {
            b.iter(|| {
                adjust_brightness_normal(&mut pixels, factor);
            })
        });
        group.bench_function(format!("normal_opt_{factor}"), |b| {
            b.iter(|| {
                adjust_brightness_normal_opt(&mut pixels, factor);
            })
        });
        group.bench_function(format!("neon_{factor}"), |b| {
            b.iter(|| {
                adjust_brightness_neon(&mut pixels, factor);
            })
        });
        group.bench_function(format!("neon_opt_{factor}"), |b| {
            b.iter(|| {
                adjust_brightness_neon_opt(&mut pixels, factor);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
