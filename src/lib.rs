use std::fs::File;
use std::io::Write;
use std::ops::{Deref, DerefMut};
use bitvec::bitvec;
use bitvec::field::BitField;
use bitvec::order::Msb0;
use bitvec::prelude::{BitSlice, BitStore, BitVec};
use bitvec::view::BitView;
use num_traits::{abs, Bounded, PrimInt, ToPrimitive, Unsigned, ToBytes};
use sha2::{Digest, Sha256};
use anyhow::{Result, anyhow, bail};
use rawloader::{RawImage, RawImageData};
use serde_json::{Map, Value};
// Pixels are variable size, up to 4 bits are used to specify the size of a pixel.
// For images with less bit-depth fewer bits can be used.

type Bits = BitVec<u8, Msb0>;

const MAGIC: u8 = 0xD6;
const VERSION: u8 = 1;

const EXIFTOOL: &str = "exiftool.exe";

/// This function compresses a single pixel and returns it representation as Bits.
/// It works by computing base_pixel - to_compress, determining the maximum number
/// of bits needed to represent this difference, truncating the difference to that
/// and then encoding the size in SIZE_WIDTH bits that prepended to the start of the
/// bitvec. If difference is 0 then size will be 0.
///
///
pub fn compress_pixel(
    base_pixel: &u64,
    to_compress: &u64,
    size_map: &Vec<u8>,
    inverse_size_map: &Vec<u8>,
    size_width: usize,
) -> Bits {
    let difference = *base_pixel as i128 - *to_compress as i128;
    let mut size: u8 = 0;

    if difference == 0 {
        return bitvec!(u8, Msb0; 0; size_width);
    }

    size = *inverse_size_map.get(get_width(difference) as usize).unwrap();

    let mut return_string = Bits::new();
    let mut size_bits = size.view_bits::<Msb0>();
    // Resize to 4 bits
    (_, size_bits) = size_bits.split_at(size_bits.len() - size_width);

    return_string.extend(size_bits);

    let binding: [u8; 16] = (difference as u128).to_be_bytes();
    let mut diff: &BitSlice<u8, Msb0> = &BitVec::from_iter(binding.iter())[..];
    // Resize to the size determined by size.
    (_, diff) = diff.split_at(diff.len() - (size_map[size as usize] as usize));
    return_string.extend(diff);
    //println!("Compressed Pixel: {:b}, base pixel: {:b}, to_compress: {:b}, size_bits: {:b}, difference bits: {:b}", return_string, *base_pixel, *to_compress, size_bits, diff);
    return_string
}

/// Decompresses a pixel from it's bit-representation to a u64.
pub fn decompress_pixel<T: BitStore>(base_pixel: &u64, to_decompress: &BitSlice<T, Msb0>) -> u64 {
    if to_decompress.count_ones() == 0 {
        return *base_pixel;
    }
    //println!("{} - {}", *base_pixel as i128, to_decompress.load_be::<i128>());
    (*base_pixel as i128 - to_decompress.load_be::<i128>()) as u64
}

/// Decompresses each pixel within the passed arrays.
/// A decompressed pixel must be able to fit within T.
/// ie if bit_depth is 14 must be at least a u16.
///
///
pub fn decompress_pixels<
    T: ToPrimitive + PrimInt + Unsigned + BitStore + bitvec::macros::internal::funty::Integral,
    V: ToPrimitive + PrimInt + Unsigned + BitStore,
>(
    base_data: &[T],
    compressed_data: &[V],
    bit_depth: usize,
    size_map: &Vec<u8>,
    size_width: usize,
) -> Vec<T> {
    let effective_bitdepth = PrimInt::trailing_ones(<T as Bounded>::max_value()) as usize;
    // We use the type of the base data for an intermediate representation of the bytes, so we need to ensure proper fitment
    assert!(effective_bitdepth >= bit_depth);
    let mut ret: Vec<T> = Vec::with_capacity(base_data.len());

    let mut base_data_idx: usize = 0;
    let mut compressed_data_idx: usize = 0;

    let base_data_bits = BitSlice::<T, Msb0>::from_slice(base_data);
    let compressed_data_bits = BitSlice::<V, Msb0>::from_slice(compressed_data);
    //println!("compressed_data_bits: {:?}", compressed_data_bits);
    while base_data_idx < base_data_bits.len() {
        // println!("Decompressing: {:?}", &compressed_data_bits[compressed_data_idx..compressed_data_idx + size_width]);
        let size_idx = compressed_data_bits[compressed_data_idx..compressed_data_idx + size_width]
            .load_be::<u8>();
        let size = *size_map.get(size_idx as usize).unwrap();
        // We don't necessarily need to use a u64 for every case here, however it will accommodate data up to a bit depth of 64 bits
        // and it's a lot simpler than switching up the type.

        let decompressed = decompress_pixel(
            &base_data_bits[base_data_idx..base_data_idx + effective_bitdepth].load_be::<u64>(),
            &compressed_data_bits[compressed_data_idx + size_width
                ..compressed_data_idx + size_width + size as usize],
        );

        //println!("Size Bits: {:b}, Compressed size: {}, decompressed: {:b}", *size_idx, size, decompressed.view_bits::<Msb0>()[64 - bit_depth..64].load_be::<T>());

        ret.push(decompressed.view_bits::<Msb0>()[64 - bit_depth..64].load_be::<T>());

        base_data_idx += effective_bitdepth;
        compressed_data_idx += size as usize + size_width;
    }
    ret
}

/// Compresses an array of pixels relative to base_data.
/// The two arrays must be of the same size.
pub fn compress_pixels<T: ToPrimitive + BitStore>(
    base_data: &[T],
    to_compress: &[T],
    bit_depth: usize,
    size_map: &Vec<u8>,
    inverse_size_map: &Vec<u8>,
    size_width: usize,
) -> Vec<T> {
    let mut intermediate: BitVec<T, Msb0> = BitVec::with_capacity(base_data.len() * bit_depth);

    for (bd, dc) in base_data.iter().zip(to_compress.iter()) {
        intermediate.extend(compress_pixel(
            bd.to_u64().as_ref().unwrap(),
            dc.to_u64().as_ref().unwrap(),
            size_map,
            inverse_size_map,
            size_width
        ));
    }

    intermediate.into_vec()
}

pub fn write_metadata(data: &str, image_file: &str, json_tmp_file: &str) -> Result<()> {
    // Write metadata to json file
    let mut f = File::create(json_tmp_file)?;
    f.write_all(data.as_bytes())?;
    f.flush()?;

    let mut exiftool = exiftool::ExifTool::with_executable(EXIFTOOL.as_ref())?;
    exiftool.execute_raw(&["-tagsFromFile", json_tmp_file, "-overwrite_original", image_file])?;
    Ok(())
}

pub fn compress_metadata_json(base_data: &Map<String, Value>, to_compress: &Map<String, Value>) -> Result<Map<String, Value>> {
    let mut to_compress_map_result = to_compress.clone();

    // Iterate over each key in to_compress and remove an entry if it exists, and it has the same value withing base_data.
    // We do this because the data is redundant if it's the same in both.
    for (to_compress_key, to_compress_value) in to_compress.iter() {
        let base_value = base_data.get(to_compress_key);

        if base_value.is_some() && base_value.unwrap() == to_compress_value {
            to_compress_map_result.remove(to_compress_key);
        }
    }


    Ok(to_compress_map_result)
}

pub fn decompress_metadata_json(base_data: &Map<String, Value>, to_decompress: &Map<String, Value>) -> Result<Map<String, Value>> {
    let mut return_value = base_data.clone();

    //TODO: Again SLOWWWWW, so much cloning, this object could be rather large.
    (&mut return_value).extend(to_decompress.iter().map(
        |(to_decompress_key, to_decompress_value)|
            {
                (to_decompress_key.clone(), to_decompress_value.clone())
            }
    ));

    Ok(return_value)
}

pub fn get_pixel_data(img: &RawImage) -> Result<&Vec<u16>> {
    match &img.data {
        RawImageData::Integer(data) => Ok(data),
        RawImageData::Float(_) => {
            panic!("Can't decode image as float");
        }
    }
}

#[derive(Debug)]
pub struct RawImageEq(RawImage);

impl PartialEq<Self> for RawImageEq {
    fn eq(&self, rhs: &Self) -> bool {
        for i in 0..self.wb_coeffs.len() {
            // NaN != NaN, I think RawImage has a parsing issue that is leading to sometimes being included in the wb_coeffs
            // It doesn't matter too much in my case considering I'm not really using the field and the actual coeffs will be
            // pulled from exiftool.
            if self.wb_coeffs[i] != rhs.wb_coeffs[i] && !self.wb_coeffs[i].is_nan() && !rhs.wb_coeffs[i].is_nan() {
                return false;
            }
        }

        if !(self.model == rhs.model && self.make == rhs.make &&
            self.blackareas == rhs.blackareas && self.crops == rhs.crops &&
            self.whitelevels == rhs.whitelevels && self.width == rhs.width &&
            self.height == rhs.height && self.cpp == rhs.cpp &&
            self.xyz_to_cam == rhs.xyz_to_cam &&
            self.cfa.name == rhs.cfa.name &&
            self.orientation == rhs.orientation) {
            return false
        }

        if get_pixel_data(self).unwrap() != get_pixel_data(rhs).unwrap() {
            return false
        }

        true
    }
}

impl Eq for RawImageEq {}

impl Deref for RawImageEq {
    type Target = RawImage;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RawImageEq {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}


#[derive(Eq, PartialEq, Debug)]
pub struct CompressedFile {
    pub header: Header,
    pub metadata: Map<String, Value>,
    pub data: Vec<u8>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct UncompressedFile {
    pub image: RawImageEq,
    bit_depth: usize,
    metadata: Map<String, Value>
}

pub struct UncompressedFileBatch([UncompressedFile]);

impl UncompressedFile {
    pub fn new(path: &str, bit_depth: usize) -> Result<UncompressedFile> {
        let image = rawloader::decode_file(path)?;

        match &image.data {
            RawImageData::Integer(_) => (),
            RawImageData::Float(_) => bail!("Uncompressed file does not support float type"),
        };

        // TODO: We need a way to retrieve the metadata in batch because it will be a good deal faster.
        let mut exiftool = exiftool::ExifTool::with_executable(EXIFTOOL.as_ref())?;
        // TODO: SLOW
        let metadata = exiftool.json(path.as_ref(), &["-b"])?.as_object().unwrap().clone();


        Ok(UncompressedFile {
            image: RawImageEq(image),
            bit_depth,
            metadata
        })
    }
}

impl CompressedFile {
    pub fn from_uncompressed(base_image: &UncompressedFile, to_compress: &UncompressedFile) -> Result<CompressedFile> {
        // Ensure the bit depths are the same and are representable by a u8.
        if base_image.bit_depth != to_compress.bit_depth || base_image.bit_depth > 255 {
            bail!("Error: Both images need to have the same bit depth and it needs to be at most 255.")
        }

        let base_data = get_pixel_data(&base_image.image)?;
        let to_compress_data = get_pixel_data(&to_compress.image)?;

        let (avg, std) = compute_stats(base_data, to_compress_data);
        let (size_width, sizemap) = generate_parameters(base_image.bit_depth, get_width(avg as i128) as usize);
        let inverse_sizemap = generate_inverse_size_map(&sizemap);

        let compressed_data = compress_pixels(base_data, to_compress_data, base_image.bit_depth, &sizemap, &inverse_sizemap, size_width);

        let compressed_metadata = compress_metadata_json(&base_image.metadata, &to_compress.metadata)?;

        Ok(Self::from_compressed_data(compressed_data, compressed_metadata, sizemap, base_image.bit_depth as u8))
    }

    pub fn write(&self, path: &str) -> Result<()> {
        let mut f = File::create(path)?;
        let c = flate2::Compression::new(6);
        flate2::write::DeflateEncoder::new(&mut f, c).write_all(&self.to_bytes())?;

        Ok(())
    }

    pub fn decompress(&self, base_image: &UncompressedFile) -> Result<UncompressedFile> {
        let new_metadata = decompress_metadata_json(&base_image.metadata, &self.metadata)?;

        let decompressed_data = decompress_pixels(
            get_pixel_data(&base_image.image)?,
            &self.data,
            base_image.bit_depth,
            &self.header.size_map,
            get_pixel_size(self.header.bit_depth) as usize
        );

        let img_ref = &base_image.image;

        // This is a lot faster than trying to clone the entire image...
        let new_image = RawImage {
            make: img_ref.make.clone(),
            model: img_ref.model.clone(),
            clean_make: img_ref.make.clone(),
            clean_model: img_ref.clean_model.clone(),
            width: img_ref.width,
            height: img_ref.height,
            cpp: img_ref.cpp,
            wb_coeffs: img_ref.wb_coeffs.clone(),
            whitelevels: img_ref.whitelevels.clone(),
            blacklevels: img_ref.blacklevels.clone(),
            xyz_to_cam: img_ref.xyz_to_cam.clone(),
            cfa: img_ref.cfa.clone(),
            crops: img_ref.crops.clone(),
            blackareas: img_ref.blackareas.clone(),
            orientation: img_ref.orientation.clone(),
            data: RawImageData::Integer(decompressed_data),
        };

        Ok(UncompressedFile {
            image: RawImageEq(new_image),
            bit_depth: base_image.bit_depth,
            metadata: new_metadata,
        })

    }

    pub fn from_compressed_data<T: PrimInt + Unsigned + ToBytes>(data: Vec<T>, metadata: Map<String, Value>, size_map: Vec<u8>, bit_depth: u8) -> CompressedFile {

        let mut data_as_bytes = Vec::with_capacity(data.len());

        for b in data.iter() {
            data_as_bytes.extend(b.to_be_bytes().as_ref());
        }

        let hash = Sha256::digest(&data_as_bytes).to_vec();

        let header = Header {
            version: VERSION,
            hash,
            bit_depth,
            size_map
        };

        CompressedFile {
            header,
            metadata,
            data: data_as_bytes,
        }
    }

    /// Byte structure is as follows:
    /// HS := 36+SIZEMAP_SZ
    ///
    /// HEADER   METADATA                DATA
    /// 0..HS    HS..HS+METADATA_SZ
    ///
    /// NOTE: Range end is exclusive.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();
        result.extend(self.header.to_bytes().iter());
        // Doing things this way could pose issues if the metadata uses a weird charset
        let meta = serde_json::to_string(&self.metadata).unwrap();
        result.extend(meta.len().to_be_bytes());
        result.extend(meta.bytes());
        // Note that this size is 8 bytes
        result.extend((self.data.len() as u64).to_be_bytes());
        result.extend(self.data.iter());
        result
    }

    pub fn from_bytes(data: &[u8]) -> Result<CompressedFile> {
        let header = Header::from_bytes(data)?;
        // data[header_sz] should point to the byte containing the size of the metadata
        let header_sz =  header.size() as usize;
        let metadata_offset = header_sz + 8;

        // Parse bytes to get the size of metadata
        let mut metadata_sz_bytes: [u8; 8] = [0; 8];
        // The size begins 8 bytes before the start of the metadata
        metadata_sz_bytes.copy_from_slice(&data[metadata_offset - 8..metadata_offset]);
        let meta_data_sz = u64::from_be_bytes(metadata_sz_bytes);

        // Parse bytes to get the size of data
        let data_offset = metadata_offset + 8 + meta_data_sz as usize;
        let mut data_sz_bytes: [u8; 8] = [0; 8];
        data_sz_bytes.copy_from_slice(&data[data_offset - 8..data_offset]);
        let data_sz = u64::from_be_bytes(data_sz_bytes);
        // TODO: Again we need to handle weird charsets.
        let meta_str = String::from_utf8(data[metadata_offset..(metadata_offset as u64 + meta_data_sz) as usize].to_vec())?;
        Ok(CompressedFile {
            header,
            // On a 32bit system this might cause problems for absurdly large images, good thing nobody uses 32 bit anymore...
            metadata: serde_json::from_str(meta_str.as_str())?,
            data: data[data_offset..(data_offset as u64 + data_sz) as usize].to_vec(),
        })
    }
}

#[derive(Eq, PartialEq, Debug)]
pub struct Header {
    pub version: u8,
    pub hash: Vec<u8>,
    pub bit_depth: u8,
    pub size_map: Vec<u8>,
}

impl Header {
    /// Header bytes are as follows:
    ///
    /// MAGIC   VERSION  HASH     BIT_DEPTH   SIZE_MAP_SZ    SIZE_MAP
    /// 0..1    1..2     2..33    33..34      34..35         35..(35+SIZE_MAP_SZ)
    ///
    /// NOTE: Range end is exclusive.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();
        result.push(MAGIC);
        result.push(VERSION);
        result.extend(self.hash.iter());
        result.push(self.bit_depth);
        result.push(self.size_map.len() as u8);
        result.extend(self.size_map.iter());
        result
    }

    pub fn size(&self) -> u8 {
        36 + self.size_map.len() as u8
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Header> {
        if bytes.len() != (35 + bytes[34]) as usize || bytes[0] != MAGIC {
            anyhow!("Error: Invalid header");
        }

        Ok(Header {
            version: bytes[1],
            hash: bytes[2..34].to_vec(),
            bit_depth: bytes[34],
            size_map: bytes[36..(36 + bytes[35] as usize)].to_vec(),
        })
    }
}

pub fn generate_parameters(bit_depth: usize, avg_diff_bits: usize) -> (usize, Vec<u8>) {
    let size_width = get_pixel_size(bit_depth as u8) as usize;
    let sizemap_size = (1 << size_width) as usize;
    assert!(sizemap_size <= bit_depth);
    assert!(avg_diff_bits <= bit_depth);
    let mut sizemap: Vec<u8> = vec![0; sizemap_size];

    // This will be the idx of the average within the sizemap.
    let ratio = avg_diff_bits as f32 / bit_depth as f32;
    let avg_idx = ((sizemap_size - 1) as f32 * ratio).ceil() as usize;
    let mut left_idx = if avg_idx == 0 { 0 } else { avg_idx - 1 };
    let mut right_idx = if avg_idx == bit_depth { bit_depth } else { avg_idx + 1 };

    let mut current_value = avg_diff_bits;
    let center_radius = (sizemap_size as f32 / 2f32).ceil() as usize / 3;
    let mut step = 1i32;

    sizemap[0] = 0;
    sizemap[avg_idx] = avg_diff_bits as u8;
    sizemap[sizemap_size - 1] = bit_depth as u8 + 1u8; // We need to add one for the sign bit.

    while right_idx < (sizemap_size - 1) {
        if right_idx == avg_idx + center_radius + 1 {
            step = ((bit_depth - current_value + 1) / (sizemap_size - right_idx)) as i32;
        }

        current_value = current_value + step as usize;

        sizemap[right_idx] = current_value as u8;
        right_idx += 1;
    }

    step = 1;
    current_value = avg_diff_bits;

    while left_idx > 0 {
        if left_idx as i32 == avg_idx as i32 - center_radius as i32 - 1i32 {
            step = ((bit_depth - current_value + 1) / (avg_idx - left_idx + 1)) as i32;
        }

        current_value = current_value - step as usize;

        sizemap[left_idx] = current_value as u8;
        left_idx -= 1;
    }

    (size_width, sizemap)
}

fn generate_inverse_size_map(size_map: &Vec<u8>) -> Vec<u8> {
    let mut inverse: Vec<u8> = Vec::new();
    let mut j = 0;
    let mut i = 0;

    while i <= (*size_map.last().unwrap()) as i32 {
        while j < size_map.len() && *size_map.get(j).unwrap() < i as u8 {
            j += 1;
        }

        inverse.insert(i as usize, j as u8);
        i += 1;
    }

    // We need to add one more element for the last index because technically when we're
    // indexing this guy we could have an index of sizemap.len()
    inverse.push(j as u8);

    inverse
}

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

        diff = abs(p1.to_i128().unwrap() - p2.to_i128().unwrap()).to_u64().unwrap();

        updated_avg = avg + (diff as f64 - avg) / (pixel_count as f64);
        variance += (diff as f64 - avg) * (diff as f64 - updated_avg);

        avg = updated_avg;
        min_diff = min_diff.min(diff);
        max_diff = max_diff.max(diff);
    }

    (avg, (variance / (pixel_count as f64 - 1f64)).sqrt())
}

// I would love to implement this generically, but at moment it's unnecessary.
#[inline]
pub fn get_width(value: i128) -> u8 {
    if value == 0i128 {
        return 0;
    }

    let mut positive = value.abs();
    let mut size = positive.ilog2() + 2;

    if positive.count_ones() == 1 && value.is_negative() {
        size -= 1;
    }

    size as u8
}

// Used to retrieve the width of the field that denotes a pixels size.
#[inline]
pub fn get_pixel_size(bit_depth: u8) -> u8 {
    match bit_depth {
        8..=16 => 3,
        17..=64 => 4,
        _ => bit_depth,
    }
}

#[cfg(test)]
mod tests {
    use num_traits::PrimInt;
    use rand::{Rng, RngCore};
    use serde_json::json;
    use super::*;

    const SIZE_MAP: [u8; 16] = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 41, 49, 57, 65];

    const INVERSE_SIZE_MAP: [u8; 66] = [
        0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
        11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
        14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15
    ];

    const SIZE_WIDTH: usize = 4;

    #[test]
    pub fn test_lossless_pixel_compression() {
        let mut randomness = rand::rng();
        let mut compressed;
        for _ in 0..1000 {
            let input: u64 = randomness.next_u64();
            let base_pixel: u64 = randomness.next_u64();
            compressed = compress_pixel(&base_pixel, &input, &Vec::from(SIZE_MAP), &Vec::from(INVERSE_SIZE_MAP), SIZE_WIDTH);
            decompress_pixel(&base_pixel, &compressed[4..]);
        }
    }

    fn generate_image_pair<T: PrimInt>(
        vec_size: usize,
        bit_depth: usize,
        similar: usize,
    ) -> (Vec<T>, Vec<T>) {
        let mut randomness = rand::rng();
        let mut base_data = Vec::new();
        let mut other_data = Vec::new();

        for i in 0..vec_size {
            let mut rand = randomness.next_u64();

            base_data.push(T::from(rand >> (64 - bit_depth)).unwrap());
            if similar != 0 {
                let modifier = randomness.random_range(0..similar);
                other_data.push(
                    T::from((rand >> (64 - bit_depth)) + modifier as u64)
                        .unwrap(),
                );
            } else {
                other_data.push(T::from(randomness.next_u64() >> (64 - bit_depth)).unwrap());
            }
        }

        (base_data, other_data)
    }

    #[test]
    pub fn test_lossless_pixels_compression() {
        for i in 2..=8 {
            let (base, to_compress) = generate_image_pair::<u8>(128 + i, i, 0);
            let (avg, std) = compute_stats(&base, &to_compress);
            let avg_diff_width = get_width(avg as i128);
            let (width, sizemap) = generate_parameters(i, avg_diff_width as usize);
            println!("width: {}", width);
            let inverse_sizemap = generate_inverse_size_map(&sizemap);
            let compressed = compress_pixels(&base, &to_compress, i, &sizemap, &inverse_sizemap, width);
            let decompressed = decompress_pixels(&base, compressed.as_slice(), i, &sizemap, width);
            assert_eq!(decompressed, to_compress);
        }

        for i in 9..=16 {
            let (base, to_compress) = generate_image_pair::<u16>(128 + i, i, 0);
            let (avg, std) = compute_stats(&base, &to_compress);
            let avg_diff_width = get_width(avg as i128);
            let (width, sizemap) = generate_parameters(i, avg_diff_width as usize);
            let inverse_sizemap = generate_inverse_size_map(&sizemap);
            let compressed = compress_pixels(&base, &to_compress, i, &sizemap, &inverse_sizemap, width);
            let decompressed = decompress_pixels(&base, compressed.as_slice(), i, &sizemap, width);
            assert_eq!(decompressed, to_compress);
        }

        for i in 17..=32 {
            let (base, to_compress) = generate_image_pair::<u32>(128 + i, i, 0);
            let (avg, std) = compute_stats(&base, &to_compress);
            let avg_diff_width = get_width(avg as i128);
            let (width, sizemap) = generate_parameters(i, avg_diff_width as usize);
            let inverse_sizemap = generate_inverse_size_map(&sizemap);
            let compressed = compress_pixels(&base, &to_compress, i, &sizemap, &inverse_sizemap, width);
            let decompressed = decompress_pixels(&base, compressed.as_slice(), i, &sizemap, width);
            assert_eq!(decompressed, to_compress);
        }

        for i in 33..=64 {
            let (base, to_compress) = generate_image_pair::<u64>(128 + i, i, 0);
            let (avg, std) = compute_stats(&base, &to_compress);
            let avg_diff_width = get_width(avg as i128);
            let (width, sizemap) = generate_parameters(i, avg_diff_width as usize);
            let inverse_sizemap = generate_inverse_size_map(&sizemap);
            let compressed = compress_pixels(&base, &to_compress, i, &sizemap, &inverse_sizemap, width);
            let decompressed = decompress_pixels(&base, compressed.as_slice(), i, &sizemap, width);
            assert_eq!(decompressed, to_compress);
        }
    }

    #[test]
    pub fn encoding_test() {
        let (base, to_compress) = generate_image_pair::<u16>(1024, 16, 0);
        let (avg, std) = compute_stats(&base, &to_compress);
        let (size_width, sizemap) = generate_parameters(16, get_width(avg as i128) as usize);
        let meta_data = serde_json::from_str::<Map<String, Value>>("
            {
                \"ISO\" : 16000,
                \"EXPOSURE\" : 30
            }
        ").unwrap();
        let compressed = CompressedFile::from_compressed_data(to_compress, meta_data, sizemap, 16);
        println!("{:?}", compressed);
        let bytes = compressed.to_bytes();
        let result = CompressedFile::from_bytes(&bytes).unwrap();
        assert_eq!(result, compressed);
    }

    #[test]
    pub fn real_world_test() {
        let base_image = UncompressedFile::new("data/2/1.NEF", 14).unwrap();
        let to_compress = UncompressedFile::new("data/2/2.NEF", 14).unwrap();
        let compressed = CompressedFile::from_uncompressed(&base_image, &to_compress).unwrap();

        let re_parsed = CompressedFile::from_bytes(&compressed.to_bytes()).unwrap();
        assert_eq!(compressed, re_parsed);

        let decompressed = compressed.decompress(&base_image).unwrap();

        assert_eq!(decompressed, to_compress);
        let uncompressed_size = (get_pixel_data(&to_compress.image).unwrap().len() * 2);

        println!("Compression Ratio: {:?}", compressed.data.len() as f32 / uncompressed_size as f32);

        // TODO: Check that decompression is correct.
    }
}