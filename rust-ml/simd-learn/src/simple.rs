use std::{arch::aarch64::*, time::Instant};

// Normal implementation - straightforward loop
#[inline(never)]
pub fn adjust_brightness_normal(pixels: &mut [u8], factor: f32) {
    // let factor_fixed = (factor * 65536.0) as u32;

    pixels.iter_mut().for_each(|pixel| {
        // let v = *pixel as u32;
        // *pixel = ((v * factor_fixed) >> 16) as u8
        *pixel = (*pixel as f32 * factor).min(255.0) as u8;
    });
}

#[inline(never)]
pub fn adjust_brightness_normal_opt(pixels: &mut [u8], factor: f32) {
    let factor_fixed = (factor * 65536.0) as u32;

    pixels.iter_mut().for_each(|pixel| {
        let v = *pixel as u32;
        *pixel = ((v * factor_fixed) >> 16).min(255) as u8
        // *pixel = (*pixel as f32 * factor).min(255.0) as u8;
    });
}

// NEON implementation - using ARM SIMD instructions
#[inline(never)]
pub fn adjust_brightness_neon(pixels: &mut [u8], factor: f32) {
    // Convert factor to fixed point (16.16) for integer arithmetic
    // let factor_fixed = (factor * 65536.0) as u2;
    let factor_vec = unsafe { vdupq_n_f32(factor) };
    let vmax = unsafe { vdupq_n_f32(255.0) }; // Maximum value for u8
    let len = pixels.len();
    let mut i = 0;

    // Process 16 pixels at a time using NEON for better throughput
    while i + 16 <= len {
        // Load 16 bytes as two 8-byte vectors
        unsafe {
            let v_pixels1 = vld1_u8(&pixels[i]);
            let v_pixels2 = vld1_u8(&pixels[i + 8]);

            // Convert to 16-bit for arithmetic
            let v_pixels_16_1 = vmovl_u8(v_pixels1);
            let v_pixels_16_2 = vmovl_u8(v_pixels2);

            // Multiply by factor using fixed-point arithmetic
            // Low parts
            let v_pixels_32_1_lo = vmovl_u16(vget_low_u16(v_pixels_16_1));
            let v_pixels_32_2_lo = vmovl_u16(vget_low_u16(v_pixels_16_2));

            let v_result_32_1_lo = vcvtq_u32_f32(vminq_f32(
                vmulq_f32(vcvtq_f32_u32(v_pixels_32_1_lo), factor_vec),
                vmax,
            ));
            let v_result_32_2_lo = vcvtq_u32_f32(vminq_f32(
                vmulq_f32(vcvtq_f32_u32(v_pixels_32_2_lo), factor_vec),
                vmax,
            ));
            // let v_result_32_1_lo = vmulq_u32(v_pixels_32_1_lo, factor_vec);
            // let v_result_32_2_lo = vmulq_u32(v_pixels_32_2_lo, factor_vec);

            // High parts
            let v_pixels_32_1_hi = vmovl_u16(vget_high_u16(v_pixels_16_1));
            let v_pixels_32_2_hi = vmovl_u16(vget_high_u16(v_pixels_16_2));
            let v_result_32_1_hi = vcvtq_u32_f32(vminq_f32(
                vmulq_f32(vcvtq_f32_u32(v_pixels_32_1_hi), factor_vec),
                vmax,
            ));
            let v_result_32_2_hi = vcvtq_u32_f32(vminq_f32(
                vmulq_f32(vcvtq_f32_u32(v_pixels_32_2_hi), factor_vec),
                vmax,
            ));

            // Combine and clamp to [0, 255]
            let v_result_16_1 =
                vcombine_u16(vmovn_u32(v_result_32_1_lo), vmovn_u32(v_result_32_1_hi));
            let v_result_16_2 =
                vcombine_u16(vmovn_u32(v_result_32_2_lo), vmovn_u32(v_result_32_2_hi));

            // Convert back to u8
            let v_result_u8_1 = vmovn_u16(v_result_16_1);
            let v_result_u8_2 = vmovn_u16(v_result_16_2);

            // Store result
            vst1_u8(&mut pixels[i], v_result_u8_1);
            vst1_u8(&mut pixels[i + 8], v_result_u8_2);
        }
        i += 16;
    }

    // Process 8 pixels at a time if remaining
    while i + 8 <= len {
        unsafe {
            // Load 8 bytes
            let v_pixels = vld1_u8(&pixels[i]);

            // Convert to 16-bit for arithmetic
            let v_pixels_16 = vmovl_u8(v_pixels);

            // Multiply by factor using fixed-point arithmetic
            // Low parts
            let v_pixels_32_lo = vmovl_u16(vget_low_u16(v_pixels_16));

            let v_result_32_lo =
                vcvtq_u32_f32(vmulq_f32(vcvtq_f32_u32(v_pixels_32_lo), factor_vec));

            // High parts
            let v_pixels_32_hi = vmovl_u16(vget_high_u16(v_pixels_16));
            let v_result_32_hi = vcvtq_u32_f32(vminq_f32(
                vmulq_f32(vcvtq_f32_u32(v_pixels_32_hi), factor_vec),
                vmax,
            ));

            // Shift right by 16 bits to convert back from fixed-point
            let v_result_16_lo = vmovn_u32(v_result_32_lo);
            let v_result_16_hi = vmovn_u32(v_result_32_hi);

            // Combine and clamp to [0, 255]
            let v_result_16 = vcombine_u16(v_result_16_lo, v_result_16_hi);

            // Convert back to u8
            let v_result_u8 = vmovn_u16(v_result_16);

            // Store result
            vst1_u8(&mut pixels[i], v_result_u8);
        }

        i += 8;
    }

    // Handle remaining pixels with scalar approach
    while i < len {
        pixels[i] = (pixels[i] as f32 * factor).min(255.0) as u8;
        i += 1;
    }
}

// NEON implementation - using ARM SIMD instructions
#[inline(never)]
pub fn adjust_brightness_neon_opt(pixels: &mut [u8], factor: f32) {
    // Convert factor to fixed point (16.16) for integer arithmetic
    let factor_fixed = (factor * 65536.0) as u32;
    let factor_vec = unsafe { vdupq_n_u32(factor_fixed) };
    let vmax = unsafe { vdupq_n_u16(255) }; // Maximum value for u8
    let len = pixels.len();
    let mut i = 0;

    // Process 16 pixels at a time using NEON for better throughput
    while i + 16 <= len {
        // Load 16 bytes as two 8-byte vectors
        unsafe {
            let v_pixels1 = vld1_u8(&pixels[i]);
            let v_pixels2 = vld1_u8(&pixels[i + 8]);

            // Convert to 16-bit for arithmetic
            let v_pixels_16_1 = vmovl_u8(v_pixels1);
            let v_pixels_16_2 = vmovl_u8(v_pixels2);

            // Multiply by factor using fixed-point arithmetic
            // Low parts
            let v_pixels_32_1_lo = vmovl_u16(vget_low_u16(v_pixels_16_1));
            let v_pixels_32_2_lo = vmovl_u16(vget_low_u16(v_pixels_16_2));
            let v_result_32_1_lo = vmulq_u32(v_pixels_32_1_lo, factor_vec);
            let v_result_32_2_lo = vmulq_u32(v_pixels_32_2_lo, factor_vec);

            // High parts
            let v_pixels_32_1_hi = vmovl_u16(vget_high_u16(v_pixels_16_1));
            let v_pixels_32_2_hi = vmovl_u16(vget_high_u16(v_pixels_16_2));
            let v_result_32_1_hi = vmulq_u32(v_pixels_32_1_hi, factor_vec);
            let v_result_32_2_hi = vmulq_u32(v_pixels_32_2_hi, factor_vec);

            // Shift right by 16 bits to convert back from fixed-point
            let v_result_16_1_lo = vshrn_n_u32(v_result_32_1_lo, 16);
            let v_result_16_2_lo = vshrn_n_u32(v_result_32_2_lo, 16);
            let v_result_16_1_hi = vshrn_n_u32(v_result_32_1_hi, 16);
            let v_result_16_2_hi = vshrn_n_u32(v_result_32_2_hi, 16);

            // Combine and clamp to [0, 255]
            let v_result_16_1 = vcombine_u16(v_result_16_1_lo, v_result_16_1_hi);
            let v_result_16_2 = vcombine_u16(v_result_16_2_lo, v_result_16_2_hi);
            let v_clamped_1 = vminq_u16(v_result_16_1, vmax);
            let v_clamped_2 = vminq_u16(v_result_16_2, vmax);

            // Convert back to u8
            let v_result_u8_1 = vmovn_u16(v_clamped_1);
            let v_result_u8_2 = vmovn_u16(v_clamped_2);

            // Store result
            vst1_u8(&mut pixels[i], v_result_u8_1);
            vst1_u8(&mut pixels[i + 8], v_result_u8_2);
        }
        i += 16;
    }

    // Process 8 pixels at a time if remaining
    while i + 8 <= len {
        unsafe {
            // Load 8 bytes
            let v_pixels = vld1_u8(&pixels[i]);

            // Convert to 16-bit for arithmetic
            let v_pixels_16 = vmovl_u8(v_pixels);

            // Multiply by factor using fixed-point arithmetic
            // Low parts
            let v_pixels_32_lo = vmovl_u16(vget_low_u16(v_pixels_16));
            let v_result_32_lo = vmulq_u32(v_pixels_32_lo, factor_vec);

            // High parts
            let v_pixels_32_hi = vmovl_u16(vget_high_u16(v_pixels_16));
            let v_result_32_hi = vmulq_u32(v_pixels_32_hi, factor_vec);

            // Shift right by 16 bits to convert back from fixed-point
            let v_result_16_lo = vshrn_n_u32(v_result_32_lo, 16);
            let v_result_16_hi = vshrn_n_u32(v_result_32_hi, 16);

            // Combine and clamp to [0, 255]
            let v_result_16 = vcombine_u16(v_result_16_lo, v_result_16_hi);
            let v_clamped = vminq_u16(v_result_16, vmax);

            // Convert back to u8
            let v_result_u8 = vmovn_u16(v_clamped);

            // Store result
            vst1_u8(&mut pixels[i], v_result_u8);
        }

        i += 8;
    }

    // Handle remaining pixels with scalar approach
    while i < len {
        pixels[i] = (pixels[i] as f32 * factor).min(255.0) as u8;
        i += 1;
    }
}

pub fn run() {
    // Create a large array of pixels for benchmarking
    let pixels: Vec<u8> = (0..(1920 * 1080 * 3))
        .map(|_| rand::random::<u8>())
        .collect();
    let factor = f32::EPSILON;

    let factor_fixed = factor * ((16.0_f32).exp2());
    println!("factor_fixed: {factor_fixed}");

    let mut v1 = pixels.clone();
    let mut v2 = pixels.clone();
    let mut v3 = pixels.clone();
    let mut v4 = pixels.clone();

    // Benchmark normal implementation
    let start = Instant::now();
    adjust_brightness_normal(&mut v1, factor);
    let normal_ts = Instant::now();

    adjust_brightness_normal_opt(&mut v2, factor);
    let normal_opt_ts = Instant::now();

    // Benchmark NEON implementation
    adjust_brightness_neon(&mut v3, factor);
    let neon_ts = Instant::now();
    adjust_brightness_neon_opt(&mut v4, factor);
    let neon_opt_ts = Instant::now();

    println!(
        "Normal implementation: {:?}",
        normal_ts.duration_since(start)
    );
    println!(
        "Normal opt implementation: {:?}",
        normal_opt_ts.duration_since(normal_ts)
    );
    println!(
        "NEON implementation:   {:?}",
        neon_ts.duration_since(normal_opt_ts)
    );
    println!(
        "NEON opt implementation:   {:?}",
        neon_opt_ts.duration_since(neon_ts)
    );

    // Verify results are the same
    if v1 == v2 && v2 == v3 && v3 == v4 {
        println!("All implementations produce identical results");
    } else {
        println!("Warning: Results differ between implementations");
    }
}
