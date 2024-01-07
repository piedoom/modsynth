#![feature(generic_const_exprs)]
#![feature(map_try_insert)]

use rand::{
    seq::{IteratorRandom, SliceRandom},
    thread_rng, Rng,
};
use std::{
    any::Any,
    collections::{binary_heap::Iter, HashMap, HashSet},
    hash::Hash,
    ops::Index,
    os::windows::thread,
};
use weighted_rand::builder::*;

pub struct Synthesizer2d<T>
where
    T: Clone + Default + PartialEq + Eq + std::hash::Hash,
{
    pub samples: Vec<Grid2d<T>>,
    pub probabilities: HashMap<T, Probabilities2d<T>>,
}

impl<T> Synthesizer2d<T>
where
    T: Clone + Default + PartialEq + Eq + std::hash::Hash + std::fmt::Debug,
{
    pub fn from_sample(sample: Grid2d<T>) -> Self {
        let probabilities = find_probabilities_2d(&sample);
        Synthesizer2d {
            samples: vec![sample],
            probabilities,
        }
    }

    pub fn add_sample(&mut self, sample: Grid2d<T>) {
        let new_probabilities = find_probabilities_2d(&sample);
        self.samples.push(sample);
        for (k, new_probability) in new_probabilities {
            if let Some(prob) = self.probabilities.get_mut(&k) {
                *prob += new_probability
            }
        }
    }

    pub fn all_cell_types(&self) -> HashSet<T> {
        self.samples
            .iter()
            .flat_map(|g| g.data.clone())
            .fold(HashSet::default(), |mut acc, t| {
                acc.insert(t);
                acc
            })
    }

    /// Reduce the options of a cell to one at random. Returns None if
    /// unsolvable.
    pub fn collapse(&self, grid: &mut Grid2d<CellState<T>>) -> Option<T> {
        let rand_coords = {
            let rand = grid.random();
            if matches!(rand.data, CellState::Solved { .. }) {
                // Can't collapse, already collapsed!
                return None;
            };
            rand.coordinates
        };
        let neighbors = grid.get_neighbors(rand_coords[0], rand_coords[1]);

        // collapse inward
        // Get all possibilities by taking appropriate neighbor direction
        // choices and summing
        let mut total_probabilities: HashMap<T, usize> = HashMap::default();

        // We need to loop through some opposite directions (in the data indicies), so we'll just list
        // the pairs here (the index is the first pair, but we zip that
        let pairs = [
            // Up/Down
            2usize, // Left/Right
            3,      // Down/Up
            0,      // Right/Left
            1,
        ];

        // Get a neighbor in a relative direction
        for (neighbor, pair) in neighbors.data.iter().zip(pairs) {
            if let Some(neighbor) = neighbor {
                // Find the possibilities in the opposite direction (our original tile)
                let probabilities = match neighbor.data {
                    CellState::Solved(t) => {
                        vec![self.probabilities.get(t).unwrap().data[pair].clone()]
                    }
                    CellState::Unsolved(t) => t
                        .iter()
                        .map(|t| self.probabilities.get(t).unwrap().data[pair].clone())
                        .collect(),
                };

                // Flatten out the hashmap (add all probabilities together to
                // get a final score)
                for probability in probabilities {
                    for (data, new_count) in probability {
                        if let Some(count) = total_probabilities.get_mut(&data) {
                            *count += new_count;
                        } else {
                            total_probabilities.insert(data.clone(), new_count);
                        }
                    }
                }
            }
        }

        // Union the total probabilities with the current cell listed
        // probabilities (and also the total possibilities)
        let (x, y) = rand_coords.into();

        let mut data = match grid.get_mut(x, y).unwrap() {
            CellState::Unsolved(t) => t,
            _ => unreachable!(),
        };

        data.retain(|d| total_probabilities.contains_key(d));
        total_probabilities.retain(|d, _| data.contains(d));

        // Return early if unsolvable
        if total_probabilities.len() == 0 {
            return None;
        }

        // Finally collapse with a weighted random
        let total_probabilities_vec: Vec<_> = total_probabilities.iter().clone().collect();
        let builder = WalkerTableBuilder::new(
            total_probabilities
                .values()
                .map(|x| *x as u32)
                .collect::<Vec<_>>()
                .as_slice(),
        );
        let wa_table = builder.build();
        let (choice, _) = total_probabilities_vec[wa_table.next()];

        // Assign the new state and return
        let rand = grid.get_mut(rand_coords[0], rand_coords[1]).unwrap();
        *rand = CellState::Solved(choice.clone());
        Some(choice.clone())
    }

    pub fn synthesize(
        &self,
        width: usize,
        height: usize,
        max_iterations: usize,
    ) -> Result<Grid2d<T>, IterationError> {
        // TODO: this could potentially lead to weird behavior if there is less
        // than 2 cell types
        let default_cell_set = CellState::Unsolved(self.all_cell_types());
        let mut grid: Grid<2, CellState<T>> = Grid2d::new([width, height]);
        // Fill the grid with all possibilities as a hash set
        grid.fill(default_cell_set);
        // Collapse a random cell to start
        let cell = grid.random_mut();
        self.collapse(&mut grid);
        // Solve the rest
        match self.solve(&mut grid, max_iterations) {
            Ok(_) => {
                let data = grid
                    .data
                    .into_iter()
                    .map(|x| {
                        if let CellState::Solved(x) = x {
                            x
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                Ok(Grid2d {
                    size: [width, height],
                    data,
                })
            }
            Err(e) => Err(e),
        }
    }

    fn solve(
        &self,
        grid: &mut Grid2d<CellState<T>>,
        max_iterations: usize,
    ) -> Result<(), IterationError> {
        let mut iterations = 0;
        let mut last_grid_state: Grid2d<CellState<T>>;
        let initial_grid_state = grid.clone();

        while iterations < max_iterations {
            last_grid_state = grid.clone();
            match self.solve_iteration(grid) {
                Ok(complete) => match complete {
                    true => return Ok(()),
                    false => iterations += 1,
                },
                Err(err) => match err {
                    IterationError::Unsolveable => {
                        // // Dead end. Increase iterations and reset to the last
                        // // grid state
                        // iterations += 1;
                        // *grid = last_grid_state;
                        iterations += 1;
                        *grid = initial_grid_state.clone();
                    }
                    IterationError::NotEnoughInformation => {
                        // Collapse another random tile
                        self.collapse(grid);
                        // Continue, adding to the iterations
                        iterations += 1;
                    }
                },
            }
        }

        // If we're here, we ran out of tries and failed.
        Err(IterationError::Unsolveable)

        // let initial = grid.clone();
        // // let mut action = Action::Continue;
        // let mut iteration_count = 0;
        // let mut attempt_count = 0;

        // while attempt_count < max_attempts || action != Action::Solved {
        //     while iteration_count < max_iterations {
        //         action = match action {
        //             Action::Failure => return,
        //             Action::ChooseRandom => {
        //                 let _ = grid
        //                     .data
        //                     .iter_mut()
        //                     .filter(|x| x.len() != 1)
        //                     .choose(&mut thread_rng())
        //                     .map(|x| match x.collapse() {
        //                         Some(_) => Action::Continue,
        //                         None => Action::Failure,
        //                     });
        //                 Action::Continue
        //             }
        //             Action::Continue => {
        //                 iteration_count += 1;
        //                 self.solve_iteration(grid)
        //             }
        //             Action::Solved => return,
        //         };
        //     }
        //     if action != Action::Solved {
        //         // If not solved, reset and retry
        //         attempt_count += 1;
        //         iteration_count = 0;
        //         action = Action::Continue;
        //         *grid = initial.clone();
        //     }
        // }

        // if action != Action::Solved {
        //     panic!("unable to solve");
        // }
    }

    /// Returns `true` if solved, `false` if unsolved, and `Error` if unsolvable
    fn solve_iteration(&self, grid: &mut Grid2d<CellState<T>>) -> Result<bool, IterationError> {
        // Clone the grid so we can mutate in place without borrow issues. Would
        // be best to fix this so we don't need to do that...
        let grid_iter = grid.clone();

        let mut changed = false;

        // Find all solved tiles so that we can iterate through them. We slap on
        // the enumeration before filtering to preserve the index
        let solved = grid_iter
            .data
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t, CellState::Solved(_)));

        let solved_count = solved.clone().count();

        // If all items are solved, skip and return success with `true`
        if solved_count == grid_iter.length() {
            return Ok(true);
        }

        // Iterate through every solved cell. We'll get the current choices for
        // each cell's neighbor options, and then perform a union on that neighbor tile's options
        for (index, state) in solved {
            match state {
                CellState::Solved(data) => {
                    // get neighbor positions and discard the probability
                    // information. We'll get it again when we get each as mut,
                    // since we can't do that all at the same time because of
                    // mutability rules
                    let (x, y) = grid.index_to_coord(index);
                    let neighbor_positions = grid_iter.get_neighbors(x, y);

                    // Loop through each neighbor position
                    for (neighbor_position, probabilities) in neighbor_positions
                        .data
                        .iter()
                        // Zip the neighbors with their corresponding
                        // probabilities determined by the current solved tile.
                        // We use unwrap here, and this should not fail unless
                        // we manually set the `data` (as this does not trigger
                        // an indexing of the `probabilities`), but that is not a public
                        // property.
                        .zip(self.probabilities.get(data).unwrap().data.iter())
                    {
                        // If the neighbor position is `None`, it means that
                        // there was no neighbor and we can skip it. We
                        // don't flatten here, as the current direction is dependent
                        // on the index of the array.
                        if let Some(neighbor_position) = neighbor_position {
                            let (x, y) = neighbor_position.coordinates.into();
                            // actually get the neighbor's own choices as
                            // mut. We only care about this neighbor if it
                            // is unsolved
                            if let CellState::Unsolved(neighbor_probabilities) =
                                grid.get_mut(x, y).unwrap()
                            {
                                // track any changes. If any constraints are solved, we can
                                // see that here.
                                let old_len = neighbor_probabilities.len();
                                // Union over the neighbor probabilities and
                                // probabilities from our current cell
                                neighbor_probabilities.retain(|p| probabilities.contains_key(p));
                                let new_len = neighbor_probabilities.len();
                                if old_len != new_len {
                                    // Some constraints were solved
                                    changed = true;
                                }
                                // If we have 0 probabilities, we are in an
                                // unsolvable state and need to backtrack or
                                // restart, so return early
                                if neighbor_probabilities.is_empty() {
                                    // Unsolveable
                                    return Err(IterationError::Unsolveable);
                                }
                            }
                        }
                    }
                }
                // Solved cells have only one entry, so this should always be `CellState::Solved`
                _ => unreachable!(),
            }
        }

        // If changed, let the caller know we successfully iterated, but didn't
        // complete all constrations
        if changed {
            Ok(false)
        }
        // Otherwise, we don't have enough info to solve any more constraints,
        // and should probably attempt to collapse a random cell
        else {
            Err(IterationError::NotEnoughInformation)
        }
    }
}

fn find_probabilities_2d<T>(sample: &Grid2d<T>) -> HashMap<T, Probabilities2d<T>>
where
    T: Clone + Default + PartialEq + Eq + std::hash::Hash,
{
    let mut probabilities: HashMap<T, Probabilities2d<T>> = HashMap::default();
    for (i, cell) in sample.data.iter().enumerate() {
        let (x, y) = sample.index_to_coord(i);
        let neighbors = sample.get_neighbors(x, y);
        // get or insert into the constraints hashmap
        if !probabilities.contains_key(cell) {
            probabilities.insert(cell.clone(), Default::default());
        }
        let probability = probabilities.get_mut(cell).unwrap();

        // Iterate in all directions
        for (neighbor, probability) in neighbors.data.iter().zip(probability.data.iter_mut()) {
            if let Some(neighbor) = neighbor {
                if let Some(probability) = probability.get_mut(neighbor.data) {
                    *probability += 1;
                } else {
                    probability.insert(neighbor.data.clone().clone(), 1);
                }
            }
        }
    }
    probabilities
}

/// Used in the grid to calculate constraints and skip over solved cells
#[derive(Clone)]
enum CellState<T>
where
    T: Clone,
{
    Solved(T),
    Unsolved(HashSet<T>),
}

#[derive(Debug, Copy, Clone)]
pub enum IterationError {
    Unsolveable,
    NotEnoughInformation,
}

#[derive(Clone, Default, Debug)]
pub struct Probabilities<const U: usize, T>
where
    [HashMap<T, usize>; U * 2]: Default + Clone,
    T: Default,
{
    data: [HashMap<T, usize>; U * 2],
}

impl<const U: usize, T> std::ops::AddAssign for Probabilities<U, T>
where
    [HashMap<T, usize>; U * 2]: Default + Clone,
    T: Clone + Default + PartialEq + Eq + std::hash::Hash,
{
    fn add_assign(&mut self, rhs: Self) {
        for (left, right) in self.data.iter_mut().zip(rhs.data.iter()) {
            for (right_data, right_count) in right {
                if let Some(left_count) = left.get_mut(right_data) {
                    *left_count += right_count;
                } else {
                    left.insert(right_data.clone(), *right_count);
                }
            }
        }
    }
}

impl<const U: usize, T> Probabilities<U, T>
where
    [HashMap<T, usize>; U * 2]: Default + Clone,
    T: Default,
{
    pub fn up(&self) -> &HashMap<T, usize> {
        &self.data[0]
    }

    pub fn up_mut(&mut self) -> &mut HashMap<T, usize> {
        &mut self.data[0]
    }

    pub fn left(&self) -> &HashMap<T, usize> {
        &self.data[1]
    }

    pub fn left_mut(&mut self) -> &mut HashMap<T, usize> {
        &mut self.data[1]
    }

    pub fn down(&self) -> &HashMap<T, usize> {
        &self.data[2]
    }

    pub fn down_mut(&mut self) -> &mut HashMap<T, usize> {
        &mut self.data[2]
    }

    pub fn right(&self) -> &HashMap<T, usize> {
        &self.data[3]
    }

    pub fn right_mut(&mut self) -> &mut HashMap<T, usize> {
        &mut self.data[3]
    }
}

#[derive(Clone)]
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

    pub fn size(&self) -> [usize; U] {
        self.size
    }
}

/// Reference to a cell
pub struct CellRef<'a, const U: usize, T> {
    pub data: &'a T,
    pub coordinates: [usize; U],
}

/// Mutable reference to a cell
pub struct CellRefMut<'a, const U: usize, T> {
    pub data: &'a mut T,
    pub coordinates: [usize; U],
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

    /// Initialize the grid with values
    pub fn fill(&mut self, data: T) {
        self.data.clear();
        for _ in 0..self.length() {
            self.data.push(data.clone())
        }
    }

    pub fn width(&self) -> usize {
        self.size[0]
    }

    pub fn height(&self) -> usize {
        self.size[1]
    }
}

impl<const U: usize, T> Grid<U, T>
where
    T: Clone + std::fmt::Display,
{
    pub fn pretty_print(&self) {
        self.data.chunks(self.width()).rev().for_each(|row| {
            for c in row {
                print!("{c}");
            }
            println!();
        });
        println!();
    }
}

impl<T> Grid2d<T>
where
    T: Clone,
{
    pub(crate) fn index_to_coord(&self, index: usize) -> (usize, usize) {
        let y = index / self.width();
        let x = index - (y * self.width());
        (x, y)
    }

    pub(crate) fn coord_to_index(&self, x: usize, y: usize) -> usize {
        (y * self.width()) + x
    }

    pub fn get(&self, x: usize, y: usize) -> Option<&T> {
        let index = self.coord_to_index(x, y);
        self.data.get(index)
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut T> {
        let index = self.coord_to_index(x, y);
        self.data.get_mut(index)
    }

    pub fn insert(&mut self, x: usize, y: usize, data: T) {
        if let Some(grid_data) = self.get_mut(x, y) {
            *grid_data = data;
        }
    }

    pub fn get_neighbors(&self, x: usize, y: usize) -> Neighbors<2, CellRef<2, T>> {
        Neighbors::new_with_data([
            // top
            {
                let (x, y) = (x, y + 1);
                self.get(x, y).map(|data| CellRef {
                    data,
                    coordinates: [x, y],
                })
            },
            // left
            if (x as isize - 1).is_negative() {
                None
            } else {
                {
                    let (x, y) = (x - 1, y);
                    self.get(x, y).map(|data| CellRef {
                        data,
                        coordinates: [x, y],
                    })
                }
            },
            // down
            if (y as isize - 1).is_negative() {
                None
            } else {
                {
                    let (x, y) = (x, y - 1);
                    self.get(x, y).map(|data| CellRef {
                        data,
                        coordinates: [x, y],
                    })
                }
            },
            //right
            {
                let (x, y) = (x + 1, y);
                self.get(x, y).map(|data| CellRef {
                    data,
                    coordinates: [x, y],
                })
            },
        ])
    }

    /// Get a random cell reference.
    /// Panics if data is not initialized
    pub fn random(&self) -> CellRef<2, T> {
        // Get a random cell
        let (index, data) = self
            .data
            .iter()
            .enumerate()
            .choose(&mut thread_rng())
            .unwrap();
        // Convert the index to coordinates
        CellRef {
            data,
            coordinates: self.index_to_coord(index).into(),
        }
    }

    /// Get a random cell mutable reference.
    /// Panics if data is not initialized
    pub fn random_mut(&mut self) -> CellRefMut<2, T> {
        // Get a random cell
        let index = thread_rng().gen_range(0..self.data.len());
        let coordinates = self.index_to_coord(index).into();
        let data = self.data.iter_mut().choose(&mut thread_rng()).unwrap();
        // Convert the index to coordinates
        CellRefMut { data, coordinates }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Neighbors<const U: usize, T>
where
    [T; U * 2]:,
{
    pub data: [Option<T>; U * 2],
}

impl<const U: usize, T> Neighbors<U, T>
where
    [T; U * 2]:,
{
    pub fn new_with_data(data: [Option<T>; U * 2]) -> Self {
        Self { data }
    }
    pub fn up(&self) -> Option<&T> {
        self.data[0].as_ref()
    }
    pub fn left(&self) -> Option<&T> {
        self.data[1].as_ref()
    }
    pub fn down(&self) -> Option<&T> {
        self.data[2].as_ref()
    }
    pub fn right(&self) -> Option<&T> {
        self.data[3].as_ref()
    }
}

impl<T> Neighbors<3, T> {
    pub fn forward(&self) -> Option<&T> {
        self.data[4].as_ref()
    }
    pub fn back(&self) -> Option<&T> {
        self.data[5].as_ref()
    }
}

pub type CellRefMut2d<'a, T> = CellRefMut<'a, 2, T>;
pub type CellRefMut3d<'a, T> = CellRefMut<'a, 3, T>;

pub type CellRef2d<'a, T> = CellRef<'a, 2, T>;
pub type CellRef3d<'a, T> = CellRef<'a, 3, T>;

pub type Grid2d<T> = Grid<2, T>;
pub type Grid3d<T> = Grid<3, T>;

pub type Probabilities2d<T> = Probabilities<2, T>;

trait HashSetExt<K> {
    fn collapse(&mut self) -> Option<&K>;
}

impl<K> HashSetExt<K> for HashSet<K>
where
    K: PartialEq + Eq + std::hash::Hash,
{
    /// Reduce the options of a hashset to one at random and return it
    fn collapse(&mut self) -> Option<&K> {
        // This would fail if no items exist, instead it will return none. This
        // shouldn't happen unless there is nothing in the hashset (i.e., it was
        // uninitialized).
        // We're also using `drain` here so we can replace self with the sole
        // new hashset option. To check if something is collapsed, we can check
        // the length of the hashset data.
        match self.drain().choose(&mut thread_rng()) {
            Some(data) => {
                self.insert(data);
                self.iter().next()
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn solve_iteration() {
    //     let grid = Grid2d {
    //         size: [6, 6],
    //         data: vec![
    //             0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         ],
    //     };
    //     grid.pretty_print();

    //     let set = CellState::Unsolved(HashSet::from_iter(grid.data.clone()));

    //     let mut gen = Grid2d::new([10, 10]);
    //     gen.fill(set);

    //     let a = gen.get_mut(3, 3).unwrap();
    //     *a = CellState::Solved(1);

    //     let synth = Synthesizer2d::from_sample(grid.clone());

    //     let _ = synth.solve_iteration(&mut gen);
    //     let _ = synth.solve_iteration(&mut gen);
    //     let _ = synth.solve_iteration(&mut gen);

    //     let new_data: Vec<_> = gen
    //         .data
    //         .iter()
    //         .map(|x| match x {
    //             CellState::Solved(e) => format!("{e}"),
    //             CellState::Unsolved(_) => "X".to_string(),
    //         })
    //         .collect();
    //     let f = Grid {
    //         size: [10, 10],
    //         data: new_data,
    //     };
    //     f.pretty_print();
    // }

    #[test]
    fn generate() {
        let grid = Grid2d {
            size: [10, 16],
            data: vec![
                "🟩", "🟩", "🟩", "🟩", "🟩", "🟩", "🟩", "🟩", "🟩", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟦", "🟩", "🟩", "🟩", "🟩", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟦", "🟩", "🏠", "🟩", "🟩", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟩", "🟩", "🟨", "🟨", "🟩", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟩", "🟨", "🟨", "🟩", "🟩", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟩", "🟨", "🟩", "🟩", "🟩", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟩", "🟩", "🟩", "🟩", "🟩", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟩", "🟩", "🟩", "🟩", "🟩", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟦", "🟩", "🟩", "🟩", "🟩", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟦", "🟩", "🟩", "🟩", "🟩", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟦", "🟩", "🟩", "🟦", "🟦", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟦", "🟩", "🟦", "🟦", "🟦", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟦", "🟦", "🟦", "🟦", "🟩", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟦", "🟦", "🟦", "🟩", "🟩", "🟩", //
                "🟩", "🟩", "🟦", "🟦", "🟦", "🟩", "🟩", "🟩", "🟩", "🟩", //
                "🟩", "🟩", "🟩", "🟩", "🟩", "🟩", "🟩", "🟩", "🟩", "🟩", //
            ]
            .chunks(10)
            .rev()
            .flatten()
            .cloned()
            .collect(),
        };

        println!("Samples:");
        grid.pretty_print();

        let mut synth = Synthesizer2d::from_sample(grid);

        println!("Generated:");

        synth.synthesize(15, 15, 10000).unwrap().pretty_print();
        synth.synthesize(6, 6, 1000).unwrap().pretty_print();
        synth.synthesize(7, 7, 1000).unwrap().pretty_print();
        synth.synthesize(8, 8, 1000).unwrap().pretty_print();
    }

    #[test]
    fn get_2d_index_to_coord() {
        let grid = Grid2d {
            size: [3, 3],
            data: vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        };

        assert_eq!(grid.index_to_coord(3), (0, 1));
        assert_eq!(grid.index_to_coord(4), (1, 1));
        assert_eq!(grid.index_to_coord(1), (1, 0));
    }

    #[test]
    fn get_2d() {
        let grid = Grid2d {
            size: [3, 2],
            data: vec!["a1", "a2", "a3", "b1", "b2", "b3"],
        };

        assert_eq!(grid.get(1, 1), Some(&"b2"));
        assert_eq!(grid.get(0, 0), Some(&"a1"));
        assert_eq!(grid.get(2, 3), None);
    }

    #[test]
    fn get_neighbors_2d() {
        let grid = Grid2d {
            size: [3, 2],
            data: vec!["a1", "a2", "a3", "b1", "b2", "b3"],
        };

        // assert_eq!(
        //     grid.get_neighbors(1, 1),
        //     Neighbors::new_with_data([None, Some(&"b1"), Some(&"a2"), Some(&"b3")])
        // );
    }

    #[test]
    fn it_works() {
        let grid = Grid::<3, usize>::new([16; 3]);
    }
}
