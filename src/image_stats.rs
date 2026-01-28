/**
  * Most of these are just useful utilities for computing image statistics.
  * The only function used within the codebase is `compute_stats`
  *
**/


use num_traits::{abs, PrimInt, ToPrimitive};
use rawloader::*;
use AstroImageCompressor::{generate_inverse_size_map, get_width};

pub fn compute_image_stats(base_image: &RawImage, others: &[RawImage]) {
    let mut total_avg: f64 = 0f64;
    let mut total_variance: f64 = 0f64;
    let mut total_min_diff: u64 = 0;
    let mut total_max_diff: u64 = 0;
    let mut total_pixel_count: u64 = 0;
    let mut img_count = 0;

    let mut diff: u64 = 0;
    let mut updated_avg: f64 = 0f64;

    let base_data = match &base_image.data {
        RawImageData::Integer(img) => img,
        RawImageData::Float(_) => {
            panic!("Cannot compute image stats from floats");
        }
    };

    for img in others {
        let mut avg: f64 = 0f64;
        let mut pixel_count: u64 = 0;
        let mut variance: f64 = 0f64;
        let mut min_diff: u64 = 0;
        let mut max_diff: u64 = 0;
        let mut compressed_size_bits: f64 = 0f64;

        img_count += 1;

        let second_data = match &img.data {
            RawImageData::Integer(img) => img,
            RawImageData::Float(_) => {
                panic!("Cannot compute image stats from floats");
            }
        };

        for (p1, p2) in base_data.iter().zip(second_data.iter()) {
            // Compute per-image stats
            pixel_count += 1;

            diff = p1.abs_diff(*p2) as u64;

            updated_avg = avg + (diff as f64 - avg) / (pixel_count as f64);
            variance += (diff as f64 - avg) * (diff as f64 - updated_avg);

            avg = updated_avg;
            min_diff = min_diff.min(diff);
            max_diff = max_diff.max(diff);

            if diff != 0 {
                compressed_size_bits += ((diff as i32).ilog2() + 2) as f64;
            }

            compressed_size_bits += 3f64;

            // Compute overall stats
            total_pixel_count += 1;
            updated_avg = total_avg + (diff as f64 - total_avg) / (total_pixel_count as f64);
            total_variance += (diff as f64 - total_avg) * (diff as f64 - updated_avg);

            total_avg = updated_avg;
            total_min_diff = total_min_diff.min(min_diff);
            total_max_diff = total_max_diff.max(max_diff);
        }

        println!("----- Base Image vs Image {} -----\nAverage Difference: {},\nTotal Pixels: {},\nMinimum Difference: {},\nMaximum Difference: {},\nStandard Deviation: {},\nCurrent Size: {}Mb,\nCompressed Size: {}Mb,\n\n", img_count, avg, pixel_count, min_diff, max_diff, (variance / (pixel_count as f64 - 1f64)).sqrt(), (img.width * img.height * 14) as f64 / (8f64 * 1024f64 * 1024f64), compressed_size_bits / (8f64 * 1024f64 * 1024f64));

    }

    println!("----- Total Statistics -----\nAverage Difference: {},\nMinimum Difference: {},\nMaximum Difference: {},\nStandard Deviation: {}\n", total_avg, total_min_diff, total_max_diff, (total_variance / (total_pixel_count as f64 - 1f64)).sqrt());

}

/// Computes the average and standard deviation of the difference in pixel values
/// between the provided images. Returns as the tuple: (avg, std)
pub fn compute_stats<T: PrimInt>(base_data: &[T], other: &[T]) -> (f64, f64) {
    let mut avg: f64 = 0f64;
    let mut pixel_count: u64 = 0;
    let mut variance: f64 = 0f64;
    let mut min_diff: u64 = 0;
    let mut max_diff: u64 = 0;
    let mut diff: u64 = 0;
    let mut updated_avg: f64 = 0f64;

    for (p1, p2) in base_data.iter().zip(other.iter()) {
        pixel_count += 1;

        diff = abs(p1.to_i64().unwrap() - p2.to_i64().unwrap()).to_u64().unwrap();

        updated_avg = avg + (diff as f64 - avg) / (pixel_count as f64);
        variance += (diff as f64 - avg) * (diff as f64 - updated_avg);

        avg = updated_avg;
        min_diff = min_diff.min(diff);
        max_diff = max_diff.max(diff);


    }

    (avg, (variance / (pixel_count as f64 - 1f64)).sqrt())
}

pub fn compute_best_possible_compression_ratio(base_image: &[u16], to_compress: &[u16], sizemap: &Vec<u8>, bit_depth: u8) {
    let mut sum = 0u64;
    let overhead = sizemap.len().ilog2();
    let inverse_size_map = generate_inverse_size_map(sizemap);

    for i in 0..base_image.len() {
        let diff = to_compress[i] as i128 - base_image[i] as i128;
        sum += sizemap[*inverse_size_map.get(get_width(diff) as usize).unwrap() as usize] as u64 + overhead as u64;
    }

    let compressed_bytes = sum as f32 / 8f32;
    let uncompressed_bytes = (to_compress.len() as f32 * bit_depth as f32) / 8f32;

    println!("Best possible compression ratio: {},", compressed_bytes / uncompressed_bytes);
}

pub fn separate_channels(data: &[u16], width: usize, height: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut r = Vec::new();
    let mut g = Vec::new();
    let mut b = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let value = data[idx];

            match (x % 2, y % 2) {
                (0, 0) => r.push(value),
                (0, 1) => g.push(value),
                (1, 0) => g.push(value),
                (1, 1) => b.push(value),
                _ => unreachable!(),
            }
        }
    }

    (r, g, b)
}