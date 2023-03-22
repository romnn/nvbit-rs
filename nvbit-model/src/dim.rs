use serde::{Deserialize, Serialize};

/// 3-dimensional coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim {
    #[must_use]
    #[inline]
    pub fn size(&self) -> u64 {
        u64::from(self.x) * u64::from(self.y) * u64::from(self.z)
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

/// Iterates over 3-dimensional coordinates.
#[derive(Debug)]
pub struct Iter {
    dim: Dim,
    current: u64,
}

impl Iterator for Iter {
    type Item = Dim;

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_lossless)]
    fn next(&mut self) -> Option<Self::Item> {
        let Self { current, dim } = *self;
        if current >= dim.size() {
            return None;
        }
        let x = current / (dim.y * dim.z) as u64;
        let yz = current % (dim.y * dim.z) as u64;
        let y = yz / dim.z as u64;
        let z = yz % dim.z as u64;
        self.current += 1;
        Some(Dim {
            x: x as u32,
            y: y as u32,
            z: z as u32,
        })
    }
}

impl IntoIterator for Dim {
    type Item = Dim;
    type IntoIter = Iter;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            dim: self,
            current: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Dim;

    #[test]
    fn test_iter_3_1_1() {
        let dim: Dim = Dim::from((3, 1, 1));
        assert_eq!(
            dim.into_iter().map(|d| (d.x, d.y, d.z)).collect::<Vec<_>>(),
            vec![(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        );
    }

    #[test]
    fn test_iter_3_3_1() {
        let dim: Dim = Dim::from((3, 3, 1));
        assert_eq!(
            dim.into_iter().map(|d| (d.x, d.y, d.z)).collect::<Vec<_>>(),
            vec![
                (0, 0, 0),
                (0, 1, 0),
                (0, 2, 0),
                (1, 0, 0),
                (1, 1, 0),
                (1, 2, 0),
                (2, 0, 0),
                (2, 1, 0),
                (2, 2, 0)
            ]
        );
    }

    #[test]
    fn test_iter_1_3_1() {
        let dim: Dim = Dim::from((1, 3, 1));
        assert_eq!(
            dim.into_iter().map(|d| (d.x, d.y, d.z)).collect::<Vec<_>>(),
            vec![(0, 0, 0), (0, 1, 0), (0, 2, 0),]
        );
    }

    #[test]
    fn test_iter_3_1_3() {
        let dim: Dim = Dim::from((3, 1, 3));
        assert_eq!(
            dim.into_iter().map(|d| (d.x, d.y, d.z)).collect::<Vec<_>>(),
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
