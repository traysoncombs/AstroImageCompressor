
use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use rawloader::RawImageData;
use AstroImageCompressor::compress_pixels;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut c2 = c.benchmark_group("ALL");
    c2.sample_size(10);
    let base_image = rawloader::decode_file("data/1.NEF").expect("Can't decode base image");
    let to_compress = rawloader::decode_file("data/2.NEF").expect("Can't decode base image");
    //image_stats::compute_image_stats(&base_image, img_vec.as_slice());
    let base_data = &match base_image.data {
        RawImageData::Integer(data) => data,
        RawImageData::Float(_) => {
            panic!("Can't decode image as float");
        }
    }[0..1_000_000];
    let data_to_compress = &match &to_compress.data {
        RawImageData::Integer(data) => data,
        RawImageData::Float(_) => {
            panic!("Can't decode image as float");
        }
    }[0..1_000_000];

    c2.bench_function(
        "Compress Benchmark",
        |b| {
            b.iter(|| compress_pixels(&base_data, &data_to_compress, black_box(14)));
        }
    );

    let compressed_data = compress_pixels(&base_data, &data_to_compress, 14);

    c2.bench_with_input(
        BenchmarkId::new("Decompress Benchmark", 1),
        &compressed_data,
        |b, i| {
            b.iter( || compress_pixels(&base_data, &data_to_compress, black_box(14)));
        }
    );
}

pub fn size_benchmark(c: &mut Criterion) {
    let base_image = rawloader::decode_file("data/1.NEF").expect("Can't decode base image");
    let to_compress = rawloader::decode_file("data/2.NEF").expect("Can't decode base image");
    //image_stats::compute_image_stats(&base_image, img_vec.as_slice());
    let base_data = &match base_image.data {
        RawImageData::Integer(data) => data,
        RawImageData::Float(_) => {
            panic!("Can't decode image as float");
        }
    }[0..1_000_000];
    let data_to_compress = &match &to_compress.data {
        RawImageData::Integer(data) => data,
        RawImageData::Float(_) => {
            panic!("Can't decode image as float");
        }
    }[0..1_000_000];

    let compressed_data = compress_pixels(&base_data, &data_to_compress, 14);

    let mut group: BenchmarkGroup<_> = c.benchmark_group("Array Size");

    let ratio = (compressed_data.len() as f32) / (data_to_compress.len() as f32);

    println!("Compression ratio of {:?}", (compressed_data.len() as f32) / (data_to_compress.len() as f32));

}

criterion_group!(benches, criterion_benchmark, size_benchmark);
criterion_main!(benches);