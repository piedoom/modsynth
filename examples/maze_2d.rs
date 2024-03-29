#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use modsynth::{Grid2, Synthesizer2};

fn main() {
    let grid = Grid2::new_with_data(
        [17, 9],
        [
            "┌", "─", "─", "─", "┬", "─", "─", "─", "┬", "─", "─", "─", "─", "─", "─", "─",
            "┐", //
            "├", "─", "╴", " ", "│", " ", "╶", "─", "┤", " ", "╶", "─", "┐", " ", "╶", "─",
            "┤", //
            "│", " ", "┌", "─", "┤", " ", "╷", " ", "└", "─", "┐", " ", "├", "─", "╴", " ",
            "│", //
            "│", " ", "│", " ", "╵", " ", "└", "─", "┐", " ", "╵", " ", "│", " ", "╶", "─",
            "┤", //
            "│", " ", "└", "─", "─", "─", "┐", " ", "└", "─", "┬", "─", "┴", "─", "┐", " ",
            "│", //
            "├", "─", "─", "─", "┐", " ", "└", "─", "─", "─", "┘", " ", "╷", " ", "│", " ",
            "│", //
            "├", "─", "╴", " ", "├", "─", "─", "─", "─", "─", "─", "─", "┤", " ", "│", " ",
            "│", //
            "│", " ", "╶", "─", "┘", " ", "╶", "─", "─", "─", "┐", " ", "╵", " ", "╵", " ",
            "│", //
            "└", "─", "─", "─", "─", "─", "─", "─", "─", "─", "┴", "─", "─", "─", "─", "─",
            "┘", //
        ]
        .chunks(17)
        .rev()
        .flatten()
        .cloned()
        .collect(),
    );

    println!("Samples:");
    println!("{}", grid.to_string_from_cell(|t| t.to_string()));

    let synth = Synthesizer2::from_sample(grid);

    println!("Generated:");

    let grid = synth.synthesize(36, 18, 10000, 16).unwrap();
    // dbg!(info);
    println!("{}", grid.to_string_from_cell(|t| t.to_string()));
}
