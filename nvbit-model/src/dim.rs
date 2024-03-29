use serde::{Deserialize, Serialize};

/// 3-dimensional coordinates.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    #[must_use]
    #[inline]
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    #[must_use]
    #[inline]
    pub fn size(&self) -> u64 {
        u64::from(self.x) * u64::from(self.y) * u64::from(self.z)
    }

    #[must_use]
    #[inline]
    fn as_tuple(&self) -> (&u32, &u32, &u32) {
        (&self.x, &self.y, &self.z)
    }
}

impl std::fmt::Display for Dim {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({},{},{})", self.x, self.y, self.z)
    }
}

impl From<u32> for Dim {
    #[inline]
    fn from(dim: u32) -> Self {
        Self { x: dim, y: 1, z: 1 }
    }
}

impl From<(u32, u32)> for Dim {
    #[inline]
    fn from(dim: (u32, u32)) -> Self {
        let (x, y) = dim;
        Self { x, y, z: 1 }
    }
}

impl From<(u32, u32, u32)> for Dim {
    #[inline]
    fn from(dim: (u32, u32, u32)) -> Self {
        let (x, y, z) = dim;
        Self { x, y, z }
    }
}

impl From<Point> for Dim {
    #[inline]
    fn from(p: Point) -> Self {
        let Point { x, y, z, .. } = p;
        Self { x, y, z }
    }
}

impl PartialEq<Point> for Dim {
    fn eq(&self, other: &Point) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

impl Ord for Dim {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_tuple().cmp(&other.as_tuple())
    }
}

impl PartialOrd for Dim {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Point {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_tuple().cmp(&other.as_tuple())
    }
}

impl PartialOrd for Point {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Iterates over 3-dimensional coordinates.
#[derive(Debug, Clone)]
pub struct Iter {
    bounds: Dim,
    current: u64,
}

/// 3-dimensional coordinates within 3 dimensional bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Point {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub bounds: Dim,
}

impl std::fmt::Display for Point {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({},{},{})", self.x, self.y, self.z)
    }
}

impl Point {
    /// Returns the unique id in the block
    #[must_use]
    #[inline]
    pub fn id(&self) -> u64 {
        let yz = u64::from(self.bounds.y) * u64::from(self.bounds.z);
        let z = u64::from(self.bounds.z);
        u64::from(self.x) * yz + u64::from(self.y) * z + u64::from(self.z)
    }

    #[must_use]
    #[inline]
    pub fn size(&self) -> u64 {
        self.bounds.size()
    }

    #[must_use]
    #[inline]
    pub fn as_tuple(&self) -> (&u32, &u32, &u32) {
        (&self.x, &self.y, &self.z)
    }

    #[must_use]
    #[inline]
    pub fn into_tuple(self) -> (u32, u32, u32) {
        self.into()
    }
}

impl From<Point> for (u32, u32, u32) {
    fn from(dim: Point) -> Self {
        (dim.x, dim.y, dim.z)
    }
}

impl Iter {
    #[must_use]
    #[inline]
    pub fn size(&self) -> u64 {
        self.bounds.size()
    }
}

impl Iterator for Iter {
    type Item = Point;

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_lossless)]
    fn next(&mut self) -> Option<Self::Item> {
        let Self { current, bounds } = *self;
        if current >= bounds.size() {
            return None;
        }
        let x = current / (bounds.y * bounds.z) as u64;
        let yz = current % (bounds.y * bounds.z) as u64;
        let y = yz / bounds.z as u64;
        let z = yz % bounds.z as u64;
        self.current += 1;
        Some(Point {
            x: x as u32,
            y: y as u32,
            z: z as u32,
            bounds,
        })
    }
}

impl IntoIterator for Dim {
    type Item = Point;
    type IntoIter = Iter;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            bounds: self,
            current: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Dim, Point};
    use pretty_assertions as diff;

    #[test]
    fn test_zero_constant() {
        let dim: Dim = Dim::ZERO;
        diff::assert_eq!(dim.size(), 0);
    }

    mod sort {
        use super::{diff, Dim, Point};

        #[test]
        fn test_sort_3_1_1() {
            let dim: Dim = Dim::from((3, 1, 1));
            diff::assert_eq!(
                dim.into_iter().map(Point::into_tuple).collect::<Vec<_>>(),
                vec![(0, 0, 0), (1, 0, 0), (2, 0, 0)]
            );
        }

        #[test]
        fn test_sort_2_2_2() {
            let dim: Dim = Dim::from((2, 2, 2));
            diff::assert_eq!(
                dim.into_iter().map(Point::into_tuple).collect::<Vec<_>>(),
                vec![
                    (0, 0, 0),
                    (0, 0, 1),
                    (0, 1, 0),
                    (0, 1, 1),
                    (1, 0, 0),
                    (1, 0, 1),
                    (1, 1, 0),
                    (1, 1, 1)
                ]
            );
        }
    }

    mod iter {
        use super::{diff, Dim, Point};

        #[test]
        fn test_iter_3_1_1() {
            let dim: Dim = Dim::from((3, 1, 1));
            diff::assert_eq!(
                dim.into_iter().map(Point::into_tuple).collect::<Vec<_>>(),
                vec![(0, 0, 0), (1, 0, 0), (2, 0, 0)]
            );
        }

        #[test]
        fn test_iter_3_3_1() {
            let dim: Dim = Dim::from((3, 3, 1));
            let dim_iter = dim.into_iter();
            assert_eq!(dim_iter.size(), 9);
            diff::assert_eq!(
                dim_iter
                    .map(|d| ((d.x, d.y, d.z), d.id()))
                    .collect::<Vec<_>>(),
                vec![
                    ((0, 0, 0), 0),
                    ((0, 1, 0), 1),
                    ((0, 2, 0), 2),
                    ((1, 0, 0), 3),
                    ((1, 1, 0), 4),
                    ((1, 2, 0), 5),
                    ((2, 0, 0), 6),
                    ((2, 1, 0), 7),
                    ((2, 2, 0), 8)
                ]
            );
        }

        #[test]
        fn test_iter_1_3_1() {
            let dim: Dim = Dim::from((1, 3, 1));
            diff::assert_eq!(
                dim.into_iter().map(Point::into_tuple).collect::<Vec<_>>(),
                vec![(0, 0, 0), (0, 1, 0), (0, 2, 0),]
            );
        }

        #[test]
        fn test_iter_3_1_3() {
            let dim: Dim = Dim::from((3, 1, 3));
            diff::assert_eq!(
                dim.into_iter().map(Point::into_tuple).collect::<Vec<_>>(),
                vec![
                    (0, 0, 0),
                    (0, 0, 1),
                    (0, 0, 2),
                    (1, 0, 0),
                    (1, 0, 1),
                    (1, 0, 2),
                    (2, 0, 0),
                    (2, 0, 1),
                    (2, 0, 2),
                ]
            );
        }
    }
}
