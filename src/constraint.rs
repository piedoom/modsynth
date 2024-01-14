use rand::{seq::IteratorRandom, thread_rng};
use weighted_rand::builder::{NewBuilder, WalkerTableBuilder};

use crate::*;

use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    ops::{Deref, DerefMut},
};

/// A grid of potential cell types
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintGrid<const U: usize, T>
where
    T: Clone + PartialEq + Eq + Hash,
{
    /// The backing grid of [`CellState`]s
    pub data: Grid<U, CellState<T>>,
}

impl<const U: usize, T> ConstraintGrid<U, T>
where
    T: Clone + Eq + Hash,
{
    /// Get a `HashSet` of all possible cell types currently in the grid
    pub fn all_cell_types(&self) -> HashSet<T> {
        let mut init: HashSet<T, _> = HashSet::default();
        for state in self.data.data.iter() {
            for kind in &state.data {
                init.insert(kind.clone());
            }
        }
        init
    }
}

impl<const U: usize, T> ConstraintGrid<U, T>
where
    T: Clone + PartialEq + Eq + Hash,
{
    /// Create a new uninitialized constraint grid
    pub fn new(size: [usize; U]) -> Self {
        Self {
            data: Grid::new(size),
        }
    }
    /// Create a new `ConstraintGrid` and initialize with some data
    pub fn new_with_data(size: [usize; U], data: Vec<CellState<T>>) -> Self {
        // Ensure each grid dimension is at least of size 1
        let mut s = Self::new(size);
        s.data.data = data;
        s
    }

    /// Returns true if all cells in this grid are solved
    pub fn is_solved(&self) -> bool {
        self.get_solved().count() == self.length()
    }

    /// Get alls solved cells
    pub fn get_solved(&self) -> impl Iterator<Item = CellRef<U, CellState<T>>> {
        self.data.iter().filter(|x| x.data.is_solved())
    }

    /// Get all unsolved cells
    pub fn get_unsolved(&self) -> impl Iterator<Item = CellRef<U, CellState<T>>> {
        self.data
            .iter()
            .filter(|x| !x.data.is_solved() && !x.data.impossible())
    }
}

impl<const U: usize, T> Deref for ConstraintGrid<U, T>
where
    T: Clone + PartialEq + Eq + Hash,
{
    type Target = Grid<U, CellState<T>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
impl<const U: usize, T> DerefMut for ConstraintGrid<U, T>
where
    T: Clone + PartialEq + Eq + Hash,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Used in the grid to calculate constraints and skip over solved cells
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellState<T>
where
    T: Clone + PartialEq + Eq + Hash,
{
    /// The possibilities for this cell
    pub data: HashSet<T>,
}

impl<T> CellState<T>
where
    T: Clone + PartialEq + Eq + Hash,
{
    /// Wrap a `HashSet` in the `CellState`
    pub fn new(data: HashSet<T>) -> Self {
        Self { data }
    }

    /// Whether or not this `CellState` is solved
    pub fn is_solved(&self) -> bool {
        self.data.len() == 1
    }

    /// Gets the singular solved item, if it is solved. Otherwise, returns none.
    pub fn get_solved(&self) -> Option<&T> {
        if self.data.iter().len() == 1 {
            self.data.iter().next()
        } else {
            None
        }
    }

    /// Whether or not this `CellState` is impossible and cannot be solved
    pub fn impossible(&self) -> bool {
        self.data.is_empty()
    }

    /// Return this `CellState` as its single possibility. Returns an error if
    /// there is not exactly one option left (i.e., this cell is not yet solved).
    pub fn flattened(self) -> Result<T> {
        if self.data.len() != 1 {
            return Err(Error::UnsolvedFlatten);
        }
        Ok(self.data.into_iter().next().unwrap())
    }
}

impl<T> FromIterator<T> for CellState<T>
where
    T: Clone + Eq + Hash,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            data: HashSet::from_iter(iter),
        }
    }
}

impl<T> From<T> for CellState<T>
where
    T: Clone + Eq + Hash,
{
    fn from(value: T) -> Self {
        Self {
            data: HashSet::from_iter([value]),
        }
    }
}

/// 2-dimensional [`ConstraintGrid`]
pub type ConstraintGrid2<T> = ConstraintGrid<2, T>;

impl<T> ConstraintGrid2<T>
where
    T: Default + Clone + PartialEq + Eq + Hash,
{
    /// Loop over every cell and solve at least once.
    /// Returns `Ok(true)` if completely solved.
    pub fn solve_iteration(&mut self, probabilities: &Probabilities2<T>) -> Result<bool> {
        let original_solved = self.get_solved().count();
        for id in 0..self.length() {
            self.solve_at(id, probabilities).ok();
            if self.get(id).unwrap().impossible() {
                return Err(Error::Impossible);
            }
        }
        let new_solved = self.get_solved().count();

        if original_solved == new_solved {
            Err(Error::MissingInformation)
        } else {
            Ok(self.is_solved())
        }
    }

    /// Solve the cell with the corresponding `id`.
    /// May only reduce possibilities of the cell without fully solving.
    /// Returns true if cell is completely solved
    pub fn solve_at(
        &mut self,
        id: impl Into<Id<2>>,
        probabilities: &Probabilities2<T>,
    ) -> Result<bool> {
        let id = id.into();
        // Get the mutable cell from our `id`
        let cell_ref = self.get(id).cloned();
        match cell_ref {
            Some(cell_ref) => {
                if cell_ref.is_solved() {
                    return Err(Error::AlreadySolved);
                }

                let to_intersect = self.probabilities_from_neighbors(id, probabilities);

                let cell_mut = self.get_mut(id).unwrap();
                // Intersect all neighboring determinations onto our current cell

                for intersection in to_intersect {
                    // Skip intersecting for directions that had nothing (borders)
                    if intersection.is_empty() {
                        continue;
                    }
                    cell_mut.data.retain(|x| intersection.contains_key(x));
                }

                if cell_mut.data.is_empty() {
                    Err(Error::Impossible)
                } else if cell_ref.data.len() == cell_mut.data.len() {
                    // No changes were made, so the attempt to solve this actually
                    // failed
                    Err(Error::MissingInformation)
                } else {
                    // Changes were made. Check if cell is fully solved or not
                    Ok(cell_mut.is_solved())
                }
            }
            None => Err(Error::NotExist),
        }
    }

    /// Gets probabilities for the current cell to intersect with based on
    /// neighboring cell probabilities in the direction of the specified cell
    fn probabilities_from_neighbors(
        &self,
        id: impl Into<Id<2>>,
        probabilities: &Probabilities2<T>,
    ) -> [HashMap<T, usize>; 4] {
        // Find the neighbors of the selected cell. We won't apply these
        // directly, but we'll need the positions to get corresponding probabilities.
        let neighbors = self.get_neighbors(id);

        // Iterate through each neighbor, and get the probabilities
        // matching that neighbor type. The probabilities itself is a `Neighbor`
        // type, so we need to get the side that faces the original cell, which
        // is whatever is the opposite direction (so e.g. for a neighbor that is
        // to the right of the cell, we want to get the probability in the left
        // direction [which is in the current cell's direction]).
        let pairs = [
            (0usize, 2), // Up/Down
            (1, 3),      // Left/Right
            (2, 0),      // Down/Up
            (3, 1),      // Right/Left
        ];

        let mut out = [
            HashMap::<T, usize>::default(),
            HashMap::<T, usize>::default(),
            HashMap::<T, usize>::default(),
            HashMap::<T, usize>::default(),
        ];
        // We don't actually care about the directional data past getting the
        // correct part of the `Neighbor`, and can just sum things up into a
        // `Vec`. We'll later perform a boolean intersection on the possible
        // types to find out final `HashSet`.
        neighbors
            // Iterate over every direction
            .data
            .into_iter()
            // Zip in the opposite pairs
            .zip(pairs)
            // Get each neighbor's corresponding probabilities
            .for_each(|(neighbor, (i, opposite_i))| {
                if let Some(neighbor) = neighbor {
                    // Get stored probabilities for this neighbor. These
                    // probabilities were determined when samples were ingested into
                    // the synthesizer.
                    // Note that because it is not guarenteed that the neighbor is
                    // collapsed, this fetches for all possible types.
                    for p in neighbor.data.data.iter() {
                        let probability = probabilities.get(p).unwrap();
                        let side = probability.data[opposite_i].clone();
                        if let Some(side) = side {
                            for (k, v) in side.iter() {
                                if out[i].contains_key(k) {
                                    let o = out[i].get_mut(k).unwrap();
                                    *o += 1;
                                } else {
                                    out[i].insert(k.clone(), *v);
                                }
                            }
                        }
                    }
                };
            });
        out
    }

    /// Reduce the options of a cell to one at random.
    pub fn collapse_at(&mut self, id: Id<2>, probabilities: &Probabilities2<T>) -> Result<T> {
        match self.get(id).cloned() {
            Some(cell_ref) => {
                match cell_ref.impossible() {
                    true => Err(Error::Impossible),
                    false => {
                        // Do a solve for this tile, just to make sure. Don't worry about any
                        // error info

                        // Combine all items in the Vec of hashmaps into a
                        // single hashmap. Then, we'll throw away anything in
                        // that hashmap that isn't a possibility in our current
                        // cell. Finally, we'll use our random library to get a
                        // weighted random item from the remaining choices, and
                        // finish the collapse
                        let mut weights = self
                            .probabilities_from_neighbors(id, probabilities)
                            .into_iter()
                            .fold(HashMap::<T, usize>::default(), |mut acc, rhs| {
                                for (rhs_t, rhs_count) in rhs.into_iter() {
                                    match acc.get_mut(&rhs_t) {
                                        Some(lhs_t) => {
                                            *lhs_t += rhs_count;
                                        }
                                        None => {
                                            acc.insert(rhs_t, rhs_count);
                                        }
                                    }
                                }
                                acc
                            });

                        // Intersect to get only valid possibilities
                        weights.retain(|k, _| cell_ref.data.contains(k));

                        // If there are no weights left, this is unsolveable :(
                        if weights.is_empty() {
                            return Err(Error::Impossible);
                        }

                        // We now need to keep the index in a specific order for
                        // the Walker thing, so do that now:

                        let weights: Vec<(T, usize)> = weights.into_iter().collect();

                        // Pick a random choice with weights
                        let builder = WalkerTableBuilder::new(
                            weights
                                .clone()
                                .into_iter()
                                .map(|x| x.1 as u32)
                                .collect::<Vec<_>>()
                                .as_slice(),
                        );
                        let wa_table = builder.build();
                        let (choice, _) = &weights[wa_table.next()];

                        let cell_mut = self.get_mut(id).unwrap();
                        *cell_mut = CellState::from(choice.clone());

                        Ok(choice.clone())
                    }
                }
            }
            None => Err(Error::NotExist),
        }
    }

    /// Get a random unsolved cell, preferring unsolved cells neighboring solved
    /// cells . Returns `None` if there are no more unsolved cells.
    pub fn random_unsolved(&self) -> Option<CellRef2<CellState<T>>> {
        // This isn't truely random. We'll first try choosing something from the
        // unsolved neighbors of currently solved cells to avoid generating
        // things in islands. If this fails, we'll try with any tile on the
        // board
        let unsolved_neighbor = self
            .get_solved()
            .flat_map(|neighbor| {
                self.get_neighbors(neighbor.id)
                    .data
                    .iter()
                    // Ensure we only get unsolved neighbors
                    .filter(|x| x.as_ref().map(|x| !x.data.is_solved()).unwrap_or_default())
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .choose(&mut thread_rng());

        match unsolved_neighbor {
            Some(random) => random.clone(),
            None => {
                // No adjacent unsolved neighbors found. Get a completely random
                // one instead
                self.get_unsolved().choose(&mut thread_rng())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use crate::{test::CellType, CellState, ConstraintGrid2, Grid2, Id, Neighbors, Probabilities2};

    #[test]
    fn get_solved() {
        let mut grid = ConstraintGrid2::new([3, 3]);
        grid.data.fill(CellState::from_iter([0, 1, 2]));
        let cell = grid.data.get_mut(Id::Index(1)).unwrap();
        *cell = CellState::from_iter([0]);
        let cell = grid.data.get_mut(Id::Index(4)).unwrap();
        *cell = CellState::from_iter([0]);
        assert_eq!(grid.get_solved().count(), 2);
    }

    #[test]
    fn get_unsolved() {
        let mut grid = ConstraintGrid2::new([3, 3]);
        grid.data.fill(CellState::from_iter([0, 1, 2]));
        let cell = grid.data.get_mut(Id::Index(1)).unwrap();
        *cell = CellState::from_iter([0]);
        let cell = grid.data.get_mut(Id::Index(4)).unwrap();
        *cell = CellState::from_iter([0]);
        assert_eq!(grid.get_unsolved().count(), 7);
    }

    #[test]
    fn fill_grid() {
        let mut grid = ConstraintGrid2::new([3, 3]);
        let c = CellState::from_iter(CellType::as_slice());
        grid.data.fill(c.clone());

        assert_eq!(
            grid.data,
            Grid2 {
                size: [3, 3],
                data: vec![c.clone(); 9],
            }
        );
    }

    /// Determine whether or not tiles, or an entire constraint grid is solved
    /// or not
    #[test]
    fn check_solved_unsolved() {
        let b = CellState {
            data: HashSet::from_iter(["🟥", "🟦"]),
        };
        let mut grid = ConstraintGrid2::new_with_data([3, 3], vec![b.clone(); 9]);
        assert!(!grid.is_solved());
        assert_eq!(grid.get_solved().count(), 0);

        let c = grid.get_mut(4).unwrap();
        c.data = HashSet::from_iter(["🟥"]);
        assert!(!grid.is_solved());
        assert_eq!(grid.get_solved().count(), 1);

        let d = grid.get_mut(5).unwrap();
        d.data = HashSet::from_iter([]);
        assert!(!grid.is_solved());
        assert_eq!(grid.get_unsolved().count(), 7);
        assert_eq!(grid.get_solved().count(), 1);

        let e = grid.get_mut(6).unwrap();
        e.data.retain(|a| b.data.contains(a));
        assert_eq!(e.data.len(), 2);
    }

    #[test]
    fn solve_cell() {
        // 🟦🟦🟦
        // 🟦🟦🟥
        // 🟦🟦🟦
        let probabilities = Probabilities2 {
            data: HashMap::from_iter([
                (
                    "🟥",
                    Neighbors {
                        data: [
                            Some(HashMap::from_iter([("🟦", 1)])), // Up
                            Some(HashMap::from_iter([("🟦", 1)])), // Left
                            Some(HashMap::from_iter([("🟦", 1)])), // Down
                            None,                                  // Right
                        ],
                    },
                ),
                (
                    "🟦",
                    Neighbors {
                        data: [
                            Some(HashMap::from_iter([("🟦", 4), ("🟧", 1)])), // Up
                            Some(HashMap::from_iter([("🟦", 5)])),            // Left
                            Some(HashMap::from_iter([("🟦", 4), ("🟧", 1)])), // Down
                            Some(HashMap::from_iter([("🟦", 5), ("🟧", 1)])), // Right
                        ],
                    },
                ),
            ]),
        };

        let solved = CellState::new(HashSet::from_iter(["🟦"]));
        let unsolved = CellState::new(HashSet::from_iter(["🟦", "🟥"]));

        let mut grid = ConstraintGrid2::new_with_data(
            [3, 3],
            vec![
                solved.clone(),
                solved.clone(),
                solved.clone(),
                solved.clone(),
                solved.clone(),
                unsolved,
                solved.clone(),
                solved.clone(),
                solved.clone(),
            ],
        );

        assert!(grid.solve_at(5, &probabilities).unwrap());
        assert_eq!(grid.get(5).unwrap().data, CellState::from_iter(["🟧"]).data)
    }

    #[test]
    fn probabilities_from_neighbors() {
        // Doesn't really matter what these are, we're just using it for the fn
        // and building our own probabilities.
        //
        // 🟦🟦🟦
        // 🟦🟥🟦
        // 🟦🟦🟦
        let sample = ConstraintGrid2::new_with_data(
            [3, 3],
            vec![
                CellState::new(HashSet::from_iter(["🟦"])),
                CellState::new(HashSet::from_iter(["🟦"])),
                CellState::new(HashSet::from_iter(["🟦"])),
                CellState::new(HashSet::from_iter(["🟦"])),
                CellState::new(HashSet::from_iter(["🟥"])),
                CellState::new(HashSet::from_iter(["🟦"])),
                CellState::new(HashSet::from_iter(["🟦"])),
                CellState::new(HashSet::from_iter(["🟦"])),
                CellState::new(HashSet::from_iter(["🟦"])),
            ],
        );

        let blue_prob = Neighbors {
            data: [
                Some(HashMap::from_iter([("🟥", 1)])), // Up
                Some(HashMap::from_iter([("🟧", 2)])), // Left
                Some(HashMap::from_iter([("🟨", 3)])), // Down
                Some(HashMap::from_iter([("🟩", 4)])), // Right
            ],
        };

        let red_prob = Neighbors {
            data: [
                Some(HashMap::from_iter([("🟦", 1)])), // Up
                Some(HashMap::from_iter([("🟪", 2)])), // Left
                Some(HashMap::from_iter([("🟫", 3)])), // Down
                Some(HashMap::from_iter([("⬛", 4)])), // Right
            ],
        };

        let probabilities = Probabilities2 {
            data: HashMap::from_iter([("🟦", blue_prob), ("🟥", red_prob)]),
        };

        // ⬛⬛🟦
        // ⬛🟥⏹️ Selection with neighbors
        // ⬛⬛🟦
        assert_eq!(
            sample.probabilities_from_neighbors(5, &probabilities),
            [
                HashMap::from_iter([("🟨", 3)]), // Neighbor to top (blue), with bottom data
                HashMap::from_iter([("⬛", 4)]), // Neighbor to left (red), with right data
                HashMap::from_iter([("🟥", 1)]), // Neighbor to bottom (blue), with up data
                HashMap::from_iter([])
            ]
        );
    }
}
