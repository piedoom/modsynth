use std::{
    collections::{HashMap, HashSet},
    ops::{Deref, DerefMut},
};

use crate::{CellState, ConstraintGrid2, Error, Grid2, Id, Neighbors};

/// 2-dimensional [`Probabilities`]
pub type Probabilities2<T> = Probabilities<2, T>;

/// 2-dimensional Synthesizer
pub struct Synthesizer2<T>
where
    T: Clone + Default + PartialEq + Eq + std::hash::Hash,
{
    samples: Vec<Grid2<T>>,
    probabilities: Probabilities2<T>,
}

impl<T> Synthesizer2<T>
where
    T: Clone + Default + PartialEq + Eq + std::hash::Hash + std::fmt::Debug,
{
    /// Create a synthesizer from a grid sample
    pub fn from_sample(sample: Grid2<T>) -> Self {
        let probabilities = find_probabilities_2d(&sample);
        Synthesizer2 {
            samples: vec![sample],
            probabilities,
        }
    }

    /// Add another sample to the synthesizer
    pub fn add_sample(&mut self, sample: Grid2<T>) {
        let new_probabilities = find_probabilities_2d(&sample);
        self.samples.push(sample);
        self.probabilities += new_probabilities;
    }

    /// Return a `HashSet` of all potential cell types
    pub fn all_cell_types(&self) -> HashSet<T> {
        self.samples
            .iter()
            .flat_map(|g| g.data.clone())
            .fold(HashSet::default(), |mut acc, t| {
                acc.insert(t);
                acc
            })
    }

    /// Synthesize a new grid of the given dimensions with the probabilities and
    /// constraints obtained from the samples
    pub fn synthesize(
        &self,
        width: usize,
        height: usize,
        max_iterations: usize,
    ) -> crate::error::Result<Grid2<T>> {
        // Initialize a constraint grid with all possible types
        let default_cell_set = CellState::from_iter(self.all_cell_types());
        let mut grid = ConstraintGrid2::new([width, height]);
        // Fill the grid with all possibilities as a hash set
        grid.data.fill(default_cell_set);

        // Solve the rest. `solve` will run until the grid is solved, or until
        // `max_iterations` is reached - whichever comes first.
        self.solve(&mut grid, max_iterations).map(|_| Grid2 {
            size: [width, height],
            data: grid
                .data()
                .iter()
                .map(|x| x.clone().flattened().unwrap())
                .collect(),
        })
    }

    fn solve(&self, grid: &mut ConstraintGrid2<T>, max_iterations: usize) -> crate::Result<()> {
        let mut iterations = 0;
        let initial_grid_state = grid.clone();

        // Collapse a random cell to start
        let (id, _) = grid.random_unsolved().unwrap();
        grid.collapse_at(id, &self.probabilities).unwrap();

        while iterations < max_iterations {
            iterations += 1;
            match grid.solve_iteration(&self.probabilities) {
                Ok(complete) => {
                    // If complete, return
                    if complete {
                        return Ok(());
                    }
                }
                Err(err) => match err {
                    Error::Impossible => {
                        // Dead end. Reset to the initial state.
                        // TODO: Backtracking
                        *grid = initial_grid_state.clone();
                    }
                    Error::MissingInformation => {
                        // Not enough information to solve any more constraints.
                        if let Some((id, _)) = grid.random_unsolved() {
                            // Collapse a new cell
                            if grid.collapse_at(id, &self.probabilities).is_err() {
                                // Impossible state, reset!
                                // TODO: Backtracking
                                *grid = initial_grid_state.clone();
                            }
                        } else {
                            // Happens when no unsolved, only impossible states.
                            // Reset!
                            *grid = initial_grid_state.clone();
                        }
                    }
                    _ => (),
                },
            }
        }

        // If we're here, we ran out of tries and failed.
        // print!("o");
        Err(Error::Impossible)
    }
}

/// All probabilities of tile pairings, with occurances.
/// Index by cell type `T`, which has a corresponding `Neighbors` holding the
/// possibilities and occurances in values
#[derive(Clone, Default, Debug)]
pub struct Probabilities<const U: usize, T>
where
    [HashMap<T, usize>; U * 2]: Default + Clone,
    T: Default + Clone,
{
    /// Probabilities in each direction for a corresponding cell type `T`
    pub data: HashMap<T, Neighbors<U, HashMap<T, usize>>>,
}

impl<const U: usize, T> Deref for Probabilities<U, T>
where
    T: Default + Clone,
    [std::collections::HashMap<T, usize>; U * 2]: std::default::Default,
{
    type Target = HashMap<T, Neighbors<U, HashMap<T, usize>>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<const U: usize, T> DerefMut for Probabilities<U, T>
where
    T: Default + Clone,
    [std::collections::HashMap<T, usize>; U * 2]: std::default::Default,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<const U: usize, T> std::ops::AddAssign for Probabilities<U, T>
where
    [HashMap<T, usize>; U * 2]: Default + Clone,
    T: Clone + Default + PartialEq + Eq + std::hash::Hash,
{
    fn add_assign(&mut self, rhs: Self) {
        for (data, right) in rhs.data.iter() {
            // Neighbors probabilities exists for this type
            if let Some(left) = self.get_mut(data) {
                for (left_data, right_data) in left.data.iter_mut().zip(right.data.iter()) {
                    if let (Some(left_data), Some(right_data)) = (left_data, right_data) {
                        for (key, count) in right_data {
                            if let Some(left_from_key) = left_data.get_mut(key) {
                                *left_from_key += count;
                            } else {
                                left_data.insert(key.clone(), *count);
                            }
                        }
                    }
                }
            }
            // Neighbors probabilities does not yet exist for this type
            else {
                self.insert(data.clone(), right.clone());
            }
        }
    }
}

impl<const U: usize, T> Probabilities<U, T>
where
    [HashMap<T, usize>; U * 2]: Default + Clone,
    T: Default + Clone,
{
    // pub fn up(&self) -> &HashMap<T, usize> {
    //     &self.data[0]
    // }

    // pub fn up_mut(&mut self) -> &mut HashMap<T, usize> {
    //     &mut self.data[0]
    // }

    // pub fn left(&self) -> &HashMap<T, usize> {
    //     &self.data[1]
    // }

    // pub fn left_mut(&mut self) -> &mut HashMap<T, usize> {
    //     &mut self.data[1]
    // }

    // pub fn down(&self) -> &HashMap<T, usize> {
    //     &self.data[2]
    // }

    // pub fn down_mut(&mut self) -> &mut HashMap<T, usize> {
    //     &mut self.data[2]
    // }

    // pub fn right(&self) -> &HashMap<T, usize> {
    //     &self.data[3]
    // }

    // pub fn right_mut(&mut self) -> &mut HashMap<T, usize> {
    //     &mut self.data[3]
    // }
}

fn find_probabilities_2d<T>(sample: &Grid2<T>) -> Probabilities2<T>
where
    T: Clone + Default + PartialEq + Eq + std::hash::Hash,
{
    // The probabilities for each cell type
    let mut probabilities: Probabilities2<T> = Default::default();

    // Iterate over every cell
    for (i, cell) in sample.data.iter().enumerate() {
        // Find the neighbors of the current cell
        let neighbors = sample.get_neighbors(Id::Index(i));

        // Find the probabilities for this cell type, if they exist. Otherwise,
        // insert and return a reference to a new `Neighbors` record with this
        // cell type, initialized with all `None` values.
        let cell_probabilities: &mut Neighbors<2, HashMap<T, usize>> =
            if probabilities.contains_key(cell) {
                probabilities.get_mut(cell).unwrap()
            } else {
                probabilities.insert(
                    cell.clone(),
                    Neighbors::<2, HashMap<T, usize>> {
                        data: [None, None, None, None],
                    },
                );
                probabilities.get_mut(cell).unwrap()
            };

        // Iterate over each direction of or neighbors data and zip with each
        // direction of our probability
        //
        // - `maybe_neighbor` is the thing we want to track the occurance of
        // - `maybe_probability` is the hashmap containing all occurance pairs that we
        // want to get/insert
        for (i, maybe_neighbor) in neighbors.data.iter().enumerate() {
            // If there is a neighbor in this direction...
            if let Some(neighbor) = maybe_neighbor {
                // Get or insert the neighbor probability for this
                // direction. This is `None` if there is no probability yet, and
                // we should insert one
                if let Some(probability) = &mut cell_probabilities.data[i] {
                    // Increase count if the value key is already in the
                    // hashmap for this direction
                    if let Some(count) = probability.get_mut(neighbor.data) {
                        *count += 1;
                    } else {
                        // Insert the probability and initialize with 1
                        probability.insert(neighbor.data.clone(), 1);
                    }
                } else {
                    // There was nothing registered for this cell value yet for anything.
                    // There is at least one thing, so initialize the hashmap
                    // with occurance
                    cell_probabilities.data[i] =
                        Some(HashMap::from_iter([(neighbor.data.clone(), 1)].into_iter()));
                }
            }
            // to
            else {
                // TODO: We can't do this actually. We'll need to still
                // insert a record for the times it was set to none, or
                // have the user identify a default block that equates to default
                //
                // We still need to record when there is no neighbor and insert
                // an instance for probabilities so we can determine edge pieces
            }
        }
    }

    probabilities
}

#[cfg(test)]
mod tests {
    // use super::*;

    // #[test]
    // fn get_probabilities() {
    //     let sample = Grid2 {
    //         size: [4, 4],
    //         data: vec![
    //             "🟥", "🟨", "🟥", "🟥", //
    //             "🟥", "🟥", "🟩", "🟥", //
    //             "🟩", "🟦", "🟥", "🟥", //
    //             "🟥", "🟨", "🟨", "🟥", //
    //         ],
    //     };

    //     let synth = Synthesizer2::from_sample(sample);

    //     // let mut grid = ConstraintGrid2::new([3, 3]);
    //     // let c = CellState::from_iter(CellType::as_slice());
    //     // grid.data.fill(c.clone());
    // }
}
