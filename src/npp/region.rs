/// Represents subregion of image.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Region {
    #[default]
    Full,
    Rectangle {
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    },
}

impl Region {
    /// Create new [`Region`] that covers the whole image.
    #[inline]
    pub fn full() -> Self {
        Region::Full
    }

    /// Create new partial [`Region`] with normalized width and height.
    ///
    /// If the `width` or `height` is less than 2, it will be set to 2 to produce a region that
    /// is valid when used with the NPP API.
    ///
    /// # Arguments
    ///
    /// * `topleft` - Coordinates of top left corner of the region.
    /// * `dims` - Dimensions of the region.
    #[inline]
    pub fn rectangle_normalized(topleft: (usize, usize), dims: (usize, usize)) -> Self {
        let (x, y) = topleft;
        let (width, height) = dims;
        Self::Rectangle {
            x,
            y,
            width: width.max(2),
            height: height.max(2),
        }
    }

    /// Resolve the actual values for `x`, `y`, `width` and `height` of the box, even if when it is
    /// `Region::Full`. To compute these, the outer `width` and `height` are required.
    ///
    /// # Arguments
    ///
    /// * `width` - Outer width.
    /// * `height` - Outer height.
    ///
    /// # Return value
    ///
    /// Region coordinates `x`, `y`, `width` and `height`.
    pub fn resolve_to_xywh(&self, width: usize, height: usize) -> (usize, usize, usize, usize) {
        match self {
            Region::Full => (0, 0, width, height),
            Region::Rectangle {
                x,
                y,
                width,
                height,
            } => (*x, *y, *width, *height),
        }
    }

    /// Whether or not the region is of type `Region::Full`.
    pub fn is_full(&self) -> bool {
        matches!(self, Region::Full)
    }
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Region::Full => write!(f, "[full]"),
            // This formats to something like this:
            //
            // ```
            // [x: 10, y: 10, width: 80, height: 40]
            // ```
            Region::Rectangle {
                x,
                y,
                width,
                height,
            } => write!(f, "[x: {x}, y: {y}, width: {width}, height: {height}]",),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_full() {
        assert_eq!(Region::full(), Region::Full);
        assert!(Region::full().is_full());
    }

    #[test]
    fn test_new_rectangle_normalized() {
        assert_eq!(
            Region::rectangle_normalized((1, 2), (3, 4)),
            Region::Rectangle {
                x: 1,
                y: 2,
                width: 3,
                height: 4
            }
        );
        assert_eq!(
            Region::rectangle_normalized((1, 2), (0, 1)),
            Region::Rectangle {
                x: 1,
                y: 2,
                width: 2,
                height: 2
            }
        );
        assert!(!Region::rectangle_normalized((1, 2), (3, 4)).is_full());
    }

    #[test]
    fn test_resolve_region() {
        let region = Region::Rectangle {
            x: 8,
            y: 10,
            width: 12,
            height: 16,
        };
        assert_eq!(region.resolve_to_xywh(20, 20), (8, 10, 12, 16));
    }

    #[test]
    fn test_resolve_full() {
        let region = Region::Full;
        assert_eq!(region.resolve_to_xywh(10, 20), (0, 0, 10, 20));
    }
}
