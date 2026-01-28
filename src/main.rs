mod image_stats;

use std::env::args;
use sha2::Digest;
use AstroImageCompressor::{CompressedFile, UncompressedFile, generate_inverse_size_map};
use crate::image_stats::compute_best_possible_compression_ratio;


fn main() {
    let args: Vec<String> = args().collect();
    if args.len() < 2 {
        println!("Error: must specify two files");
        return;
    }

    // Load base image and image to be compressed
    let base_image = UncompressedFile::new(&args[1], 16).unwrap();
    let to_compress = UncompressedFile::new(&args[2], 16).unwrap();

    // Compress our file and write to disk
    let compressed = CompressedFile::from_uncompressed(&base_image, &to_compress).unwrap();
    let _ = &compressed.write("test.compressed").unwrap();

    // Decompress compressed file and write to disk.
    let decompressed = &compressed.decompress(&base_image).unwrap();
    decompressed.write("test_decompressed.tiff").unwrap();

    // Hash uncompressed image
    let mut hasher = sha2::Sha256::new();
    let mut cloned_image_data = to_compress.image.clone();
    hasher.update(from_u16(cloned_image_data.as_mut()));

    let og_image_hash: Vec<u8> = hasher.finalize().to_vec();

    // Hash decompressed image
    hasher = sha2::Sha256::new();
    cloned_image_data = decompressed.image.clone();
    hasher.update(from_u16(cloned_image_data.as_mut()));

    let decompressed_hash: Vec<u8> = hasher.finalize().to_vec();

    let bit_overhead = &compressed.header.size_map.len().ilog2();

    println!("Sizemap: {:?}", &compressed.header.size_map);
    println!("Inverse Sizemap: {:?}", generate_inverse_size_map(&compressed.header.size_map));
    println!("Bits of overhead per pixel: {:?}", bit_overhead);

    assert_eq!(og_image_hash, decompressed_hash);
    println!("File hashes are equal! {:?} == {:?}", og_image_hash, decompressed_hash);

    let mut test_map = compressed.header.size_map.clone();
    test_map[1] = 2;

    compute_best_possible_compression_ratio(base_image.image.as_slice(), to_compress.image.as_slice(), &test_map, 16);

}

fn from_u16(from: &mut [u16]) -> &[u8] {
    if cfg!(target_endian = "little") {
        for byte in from.iter_mut() {
            *byte = byte.to_be();
        }
    }

    let len = from.len().checked_mul(2).unwrap();
    let ptr: *const u8 = from.as_ptr().cast();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}