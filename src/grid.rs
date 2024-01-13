use rand::{seq::IteratorRandom, thread_rng};

/// 2-dimensional [`CellRefMut`]
pub type CellRefMut2<'a, T> = CellRefMut<'a, 2, T>;
/// 3-dimensional [`CellRefMut`]
pub type CellRefMut3<'a, T> = CellRefMut<'a, 3, T>;

/// 2-dimensional [`CellRef`]
pub type CellRef2<'a, T> = CellRef<'a, 2, T>;
/// 3-dimensional [`CellRef`]
pub type CellRef3<'a, T> = CellRef<'a, 3, T>;

/// 2-dimensional [`Grid`]
pub type Grid2<T> = Grid<2, T>;
/// 3-dimensional [`Grid`]
pub type Grid3<T> = Grid<3, T>;

/// N-dimensional grid
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grid<const U: usize, T>
where
    T: Clone,
{
    pub(crate) size: [usize; U],
    /// Grid data, stored as a 1D array
    pub(crate) data: Vec<T>,
}

impl<const U: usize, T> Grid<U, T>
where
    T: Clone,
{
    /// Returns the length of the backing 1D array
    pub fn length(&self) -> usize {
        self.size.iter().product()
    }

    /// Gets the size of the grid
    pub fn size(&self) -> [usize; U] {
        self.size
    }

    /// Get data as a 1-dimensional `Vec`
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }
}

/// Reference to a cell
#[derive(Debug, Clone, Copy)]
pub struct CellRef<'a, const U: usize, T> {
    /// Reference to the cell data
    pub data: &'a T,
    /// Cell `Id`
    pub id: Id<U>,
}

/// Mutable reference to a cell
#[derive(Debug)]
pub struct CellRefMut<'a, const U: usize, T> {
    /// Mutable eference to the cell data
    pub data: &'a mut T,
    /// Cell `Id`
    pub id: Id<U>,
}

impl<const U: usize, T> Grid<U, T>
where
    T: Clone,
{
    /// Create a new uninitialized grid
    pub fn new(size: [usize; U]) -> Self {
        // Ensure each grid dimension is at least of size 1
        assert_ne!(size.iter().product::<usize>(), 0);
        let capacity = size.iter().product();
        Self {
            data: Vec::with_capacity(capacity),
            size,
        }
    }

    /// Create a grid and initialize it with some data
    pub fn new_with_data(size: [usize; U], data: Vec<T>) -> Self {
        // Ensure each grid dimension is at least of size 1
        let mut s = Self::new(size);
        s.data = data;
        s
    }

    /// Fill the grid with a value
    pub fn fill(&mut self, data: T) {
        self.data.clear();
        for _ in 0..self.length() {
            self.data.push(data.clone())
        }
    }

    /// Width of the grid
    pub fn width(&self) -> usize {
        self.size[0]
    }

    /// Height of the grid
    pub fn height(&self) -> usize {
        self.size[1]
    }
}

impl<const U: usize, T> Grid<U, T>
where
    T: Clone,
{
    /// Get a string formatted as a grid
    pub fn to_string_from_cell(&self, f: fn(&T) -> String) -> String {
        let mut out = String::new();
        self.data.chunks(self.width()).rev().for_each(|row| {
            for c in row {
                out += &f(c);
            }
            out += "\n";
        });
        out += "\n";
        out
    }
}

impl<const U: usize, T> Grid<U, T>
where
    T: Clone,
{
    fn random_index(&self) -> usize {
        (0..self.length()).choose(&mut thread_rng()).unwrap()
    }

    /// Get a random cell reference.
    /// Panics if data is not initialized
    pub fn random(&self) -> CellRef<2, T> {
        let id = self.random_index();
        let data = self.data.get(id).unwrap();

        // Convert the index to coordinates
        CellRef {
            data,
            id: Id::Index(id),
        }
    }

    /// Get a random cell mutable reference.
    /// Panics if data is not initialized
    pub fn random_mut(&mut self) -> CellRefMut<2, T> {
        let id = self.random_index();
        let data = self.data.get_mut(id).unwrap();

        // Convert the index to coordinates
        CellRefMut {
            data,
            id: Id::Index(id),
        }
    }
}

impl<T> Grid2<T>
where
    T: Clone,
{
    /// Get a cell with some `id`. Returns `None` if no cell exists at that `id`.
    pub fn get(&self, id: impl Into<Id<2>>) -> Option<&T> {
        self.data.get(id.into().as_index(self.width()))
    }

    /// Mutably get a cell with some `id`. Returns `None` if no cell exists at that `id`.
    pub fn get_mut(&mut self, id: impl Into<Id<2>>) -> Option<&mut T> {
        let width = self.width();
        self.data.get_mut(id.into().as_index(width))
    }

    /// Insert
    pub fn insert(&mut self, id: impl Into<Id<2>>, data: T) {
        let grid_data = self.get_mut(id).unwrap();
        *grid_data = data;
    }

    /// Get direct neighbors of a selected cell
    pub fn get_neighbors(&self, id: impl Into<Id<2>>) -> Neighbors<2, CellRef<2, T>> {
        let id = id.into();
        let (x, y) = id.as_coord(self.width()).into();
        Neighbors::new_with_data([
            // top
            {
                let (x, y) = (x, y + 1);
                self.get((x, y)).map(|data| CellRef {
                    data,
                    id: Id::Coord([x, y]),
                })
            },
            // left
            if (x as isize - 1).is_negative() {
                None
            } else {
                {
                    let (x, y) = (x - 1, y);
                    self.get((x, y)).map(|data| CellRef {
                        data,
                        id: Id::Coord([x, y]),
                    })
                }
            },
            // down
            if (y as isize - 1).is_negative() {
                None
            } else {
                {
                    let (x, y) = (x, y - 1);
                    self.get((x, y)).map(|data| CellRef {
                        data,
                        id: Id::Coord([x, y]),
                    })
                }
            },
            //right
            {
                let (x, y) = (x + 1, y);
                if x >= self.width() {
                    None
                } else {
                    self.get((x, y)).map(|data| CellRef {
                        data,
                        id: Id::Coord([x, y]),
                    })
                }
            },
        ])
    }
}

/// Identifier for a cell, either as a 1D index, or an ND coordinate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Id<const U: usize> {
    /// A 1-dimensional index
    Index(usize),
    /// An N-dimensional coordinate. Ensure that the coordinate exists, as it
    /// may result in incorrect values being returned otherwise.
    Coord([usize; U]),
}

impl Id<2> {
    /// Get the `Id` as an index
    pub fn as_index(&self, width: usize) -> usize {
        match self {
            Id::Index(i) => *i,
            Id::Coord(coord) => {
                let (x, y) = (coord[0], coord[1]);
                (y * width) + x
            }
        }
    }

    /// Get the `Id` as a coordinate
    pub fn as_coord(&self, width: usize) -> [usize; 2] {
        match self {
            Id::Index(i) => {
                let y = i / width;
                let x = i - (y * width);
                [x, y]
            }
            Id::Coord(coord) => *coord,
        }
    }
}

impl From<(usize, usize)> for Id<2> {
    fn from(value: (usize, usize)) -> Self {
        Id::Coord([value.0, value.1])
    }
}

impl From<usize> for Id<2> {
    fn from(value: usize) -> Self {
        Id::Index(value)
    }
}

/// Neighbors of a cell
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Neighbors<const U: usize, T>
where
    [T; U * 2]:,
{
    /// Neighbor data, where 0 = up, 1 = left, 2 = down, and 3 = right
    pub data: [Option<T>; U * 2],
}

impl<const U: usize, T> Neighbors<U, T>
where
    [T; U * 2]:,
{
    /// Create a new `Neighbors` with some data
    pub fn new_with_data(data: [Option<T>; U * 2]) -> Self {
        Self { data }
    }
    /// Get the neighbor above
    pub fn up(&self) -> Option<&T> {
        self.data[0].as_ref()
    }
    /// Get the neighbor to the left
    pub fn left(&self) -> Option<&T> {
        self.data[1].as_ref()
    }
    /// Get the neighbor below
    pub fn down(&self) -> Option<&T> {
        self.data[2].as_ref()
    }
    /// Get the neighbor to the right
    pub fn right(&self) -> Option<&T> {
        self.data[3].as_ref()
    }
}

impl<T> Neighbors<3, T> {
    /// Get the neighbor in front
    pub fn forward(&self) -> Option<&T> {
        self.data[4].as_ref()
    }
    /// Get the neighbor to the back
    pub fn back(&self) -> Option<&T> {
        self.data[5].as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_with_data(width: usize, height: usize) -> Grid2<usize> {
        let mut grid = Grid2::new([width, height]);
        grid.fill(0);
        grid.data.iter_mut().enumerate().for_each(|(i, x)| *x = i);
        grid
    }

    #[test]
    fn get_2_index_to_coord() {
        let grid = Grid2 {
            size: [3, 3],
            data: vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        };

        assert_eq!(
            Id::Index(3).as_coord(grid.width()),
            Id::Coord([0, 1]).as_coord(grid.width())
        );
        assert_eq!(
            Id::Index(4).as_coord(grid.width()),
            Id::Coord([1, 1]).as_coord(grid.width())
        );
        assert_eq!(
            Id::Index(1).as_coord(grid.width()),
            Id::Coord([1, 0]).as_coord(grid.width())
        );
        assert_eq!(
            Id::Index(0).as_coord(grid.width()),
            Id::Coord([0, 0]).as_coord(grid.width())
        );

        let grid = grid_with_data(7, 7);

        assert_eq!(
            Id::Index(52).as_coord(grid.width()),
            Id::Coord([3, 7]).as_coord(grid.width())
        );
    }

    #[test]
    fn get_2_coord_to_index() {
        let grid = Grid2 {
            size: [3, 3],
            data: vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        };

        assert_eq!(
            Id::Index(3).as_index(grid.width()),
            Id::Coord([0, 1]).as_index(grid.width())
        );
        assert_eq!(
            Id::Index(4).as_index(grid.width()),
            Id::Coord([1, 1]).as_index(grid.width())
        );
        assert_eq!(
            Id::Index(1).as_index(grid.width()),
            Id::Coord([1, 0]).as_index(grid.width())
        );
        assert_eq!(
            Id::Index(0).as_index(grid.width()),
            Id::Coord([0, 0]).as_index(grid.width())
        );

        let grid = grid_with_data(7, 7);

        assert_eq!(
            Id::Index(52).as_index(grid.width()),
            Id::Coord([3, 7]).as_index(grid.width())
        );
    }

    #[test]
    fn get_2() {
        let grid = Grid2 {
            size: [3, 2],
            data: vec!["a1", "a2", "a3", "b1", "b2", "b3"],
        };

        assert_eq!(grid.get((1, 1)), Some(&"b2"));
        assert_eq!(grid.get((0, 0)), Some(&"a1"));
        assert_eq!(grid.get((2, 3)), None);
    }

    #[test]
    fn get_neighbors_2() {
        let grid = Grid2 {
            size: [3, 2],
            data: [
                "b1", "b2", "b3", //
                "a1", "a2", "a3", //
            ]
            .chunks_exact(3)
            .rev()
            .flatten()
            .cloned()
            .collect(),
        };

        assert_eq!(
            grid.get_neighbors(Id::Coord([1, 1]))
                .data
                .iter()
                .map(|x| x.map(|x| x.data))
                .collect::<Vec<_>>(),
            vec![None, Some(&"b1"), Some(&"a2"), Some(&"b3")]
        );

        assert_eq!(
            grid.get_neighbors(2)
                .data
                .iter()
                .map(|x| x.map(|x| x.data))
                .collect::<Vec<_>>(),
            vec![Some(&"b3"), Some(&"a2"), None, None]
        );

        assert_eq!(
            grid.get_neighbors(0)
                .data
                .iter()
                .map(|x| x.map(|x| x.data))
                .collect::<Vec<_>>(),
            vec![Some(&"b1"), None, None, Some(&"a2")]
        );
    }
}
