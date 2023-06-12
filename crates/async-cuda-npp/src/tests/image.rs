pub type Pixel = [u8; 3];
pub type Image2x2 = [[Pixel; 2]; 2];
pub type Image4x4 = [[Pixel; 4]; 4];

pub const R: Pixel = [255_u8, 0_u8, 0_u8];
pub const G: Pixel = [0_u8, 255_u8, 0_u8];
pub const B: Pixel = [0_u8, 0_u8, 255_u8];

/// This is a 4 by 4 testing image that represents the hypothetical RGB flag, which looks something
/// like this:
///
/// ```text
/// .. .. .. ..
/// RR RR RR RR
/// RR GG GG RR
/// RR BB BB RR
/// RR RR RR RR
/// .. .. .. ..
/// ```
/// (It consists of a two-pixel green and blue band, wrapped in a red one-pixel border.)
///
/// Where `RR` represents a red pixel, `GG` a green one and `BB` a blue one.
pub const RGB_FLAG_RAW: Image4x4 = [
    [R, R, R, R], // Red border
    [R, G, G, R], // Green band with red border
    [R, B, B, R], // Blue band with red border
    [R, R, R, R], // Red border
];

/// This is the [`RGB_FLAG_RAW`] image with contiguous memory layout so that it can be easily put
/// into a host or device buffer.
pub const RGB_FLAG: [u8; 4 * 4 * 3] = flatten!(RGB_FLAG_RAW, 4 * 4 * 3);

/// Convenience macro to flatten a nested array to a flat array.
///
/// # Usage
///
/// ```ignore
/// let array = [
///     [1, 2, 3],
///     [4, 5, 6],
///     [7, 8, 9],
/// ];
/// assert_eq!(
///     &flatten!(array),
///     &[1, 2, 3, 4, 5, 6, 7, 8, 9],
/// );
/// ```
macro_rules! flatten {
    ($array:expr, $size:expr) => {
        unsafe { std::mem::transmute::<_, [_; $size]>($array) }
    };
}

pub(crate) use flatten;
