//! Model synthesis crate based on [Paul Merrell's algorithm](https://paulmerrell.org/model-synthesis/)
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(map_try_insert)]
#![deny(missing_docs)]

mod constraint;
mod error;
mod grid;
mod synthesizer;
#[cfg(test)]
mod test;
mod util;

pub use constraint::*;
pub use error::*;
pub use grid::*;
pub use synthesizer::*;
