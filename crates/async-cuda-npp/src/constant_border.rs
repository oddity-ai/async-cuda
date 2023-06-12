/// Represents a constant border around an image.
///
/// This is used to specify the border around an image when copying a constant border around it for
/// the purposes of letterbox resizing.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstantBorder {
    pub left: u32,
    pub top: u32,
    pub color: [u8; 3],
}

impl ConstantBorder {
    /// New constant border.
    ///
    /// # Arguments
    ///
    /// * `left` - Size of border on the left and right sides of the image in number of pixels.
    /// * `top`- Size of border on the top and bottom sides of the image in number of pixels.
    /// * `color` - Color of border (RGB).
    pub fn new(left: u32, top: u32, color: [u8; 3]) -> Self {
        Self { left, top, color }
    }

    /// New constant border with white color.
    ///
    /// # Arguments
    ///
    /// * `left` - Size of border on the left and right sides of the image in number of pixels.
    /// * `top`- Size of border on the top and bottom sides of the image in number of pixels.
    pub fn white(left: u32, top: u32) -> Self {
        Self::new(left, top, [255, 255, 255])
    }

    /// New constant border with black color.
    ///
    /// # Arguments
    ///
    /// * `left` - Size of border on the left and right sides of the image in number of pixels.
    /// * `top`- Size of border on the top and bottom sides of the image in number of pixels.
    pub fn black(left: u32, top: u32) -> Self {
        Self::new(left, top, [0, 0, 0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let border = ConstantBorder::new(1, 2, [3, 4, 5]);
        assert_eq!(border.left, 1);
        assert_eq!(border.top, 2);
        assert_eq!(border.color, [3, 4, 5]);
    }

    #[test]
    fn test_white() {
        let border = ConstantBorder::white(1, 2);
        assert_eq!(border.left, 1);
        assert_eq!(border.top, 2);
        assert_eq!(border.color, [255, 255, 255]);
    }

    #[test]
    fn test_black() {
        let border = ConstantBorder::black(1, 2);
        assert_eq!(border.left, 1);
        assert_eq!(border.top, 2);
        assert_eq!(border.color, [0, 0, 0]);
    }
}
