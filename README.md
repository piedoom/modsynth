# modsynth

`modsynth` is an implementation of [model
synthesis](https://paulmerrell.org/model-synthesis/) intended for use in interactive
applications like games. It can take a grid of arbitrary data and synthesize new
grids with local similarity. This is very similar/also known as "wave function
collapse" (wfc).

## Examples

### Input samples

```
🟩🟩❎❎❎🟩🟩🟩🟩🟩    🟪🟪🟪🟪⬜⬜🟩🟩🟩🟩
🟩❎🟦🟦🟦❎🟩🟩🟩🟩    🟪🟪🟪🟪🟪⬜⬜🟩🟩🟩
🟩❎🟦🟦🟦❎🏠🟩🟩🟩    🟪🟪🟪🟪🟪🟪⬜⬜🟩🟩
🟩❎🟦🟦❎🟩🟨🟨🟩🟪    🟪🟪🟪🟪🟪🟪🟪⬜⬜🟩
🟩❎🟦🟦❎🟨🟨🟩🟩🟩    🟪🟪🟪🟪🟪🟪🟪🟪⬜⬜
🟩❎🟦🟦❎🟨🟩🟩🟩🟩    🟧🟪🟪🟪🟪🟪🟪🟪🟪⬜
🟩❎🟦🟦❎🟩🟩🟩🟩🟩    🟧🟧🟪🟪🟪🟪🟪🟪🟪🟪
🟩❎🟦🟦🟦❎🟩🟩🟩🟩    🟨🟧🟧🟪🟪🟪🟪🟪🟪⬜
🟪❎🟫🟫🟫❎🟩🟩🟩🟩    🟨🟧🟧🟪🟪🟪🟪🟪⬜⬜
🟩❎🟫🟫🟫❎🟩❎❎🟩    🟧🟧🟪🟪🟪🟪🟪⬜⬜🟩
🟩❎🟦🟦🟦❎❎🟦🟦❎    🟧🟪🟪🟪🟪🟪⬜⬜🟩🟩
🟩❎🟦🟦🟦❎🟦🟦🟦❎    🟪🟪🟪🟪🟪⬜⬜🟩🟩🟩
🟩❎🟦🟦🟦🟦🟦🟦❎🟩    🟪🟪🟪🟪⬜⬜🟩🟩🟩🟩
🟩❎🟦🟦🟦🟦🟦❎🟩🟩    🟪🟪🟪⬜⬜🟩🟩🟩🟩🟩
🟩❎🟦🟦🟦❎❎🟩🟩🟩    🟪🟪🚃🚃🚃🚃🚃🚃🚃🟩
🟩🟩❎❎❎🟩🟩🟩🟪🟩    🟪⬜⬜🟩🟩🟩🟩🟩🟩🟩
```

### Output
```
🟪🟩🟨🟩🟪🟪🟪🟩🟩🟩❎❎❎🟩🟪🟪
🟪⬜🟩❎🟩🟩🟪⬜⬜⬜🟩🟩❎🟩🟪🚃
🟪🟪⬜🟩🟪🟪🟩🟩🟩🟩🟩🟩🟩🟨🟩🟩
🟪🟪🟪🟩🟪🟪❎🟩🟩🟩🟪⬜⬜🟩🟪🟩
🟪🚃🚃🚃🟩🟩🟩🟩🟨🟩🟩🟩🟪⬜🟩🟩
🟪🟩🟩🟩🟩🟩🟩❎🟩🟩🟩🟪🚃🚃🚃🟩
🟪🟪🟪🟩🟩🟩🟩🟩🟩🟪🟪⬜⬜⬜⬜⬜
🟪🟩🟪⬜⬜⬜⬜⬜🟩🟪🟪⬜🟩🟩🟪🟪
🟪🟪🟪🟪⬜⬜🟩🟩🟨🟩🟪🟩🟩🟩🟪🟪
🟪🟪🟪🟪🟪🟩🟩❎🟨🟩🟪❎🟩🟪🟪🟪
🟧🟧🟧🟧🟧🟪🟩🟩🟩🟩🟩🟩🟨🟧🟧🟪
🟨🟨🟨🟨🟧🟪🚃🚃🟩🟪🟪❎🟩🟪🟪🟪
🟩🟩🟨🟩🟪🟪⬜⬜🟩🟩🟩🟩🟩🟪🟪🟪
❎🟨🟧🟪🟪⬜⬜⬜🟩❎❎❎🟩🟪🟩🟪
🟩🟩🟪🟪🟪🟪🟩🟩🟩❎❎🟩🟩🟩🟩🟪
🟪🟪🟪🟪🟪🟩🟩❎🟩🟩🟩🟪🟪🟪🟪⬜
```

## Features / limitations

I want to implement everything here eventually.

- [x] synthesize from input samples (no need to specify constraints manually)
- [x] multiple sample support
    - [ ] multiple sample weight adjustments
- [ ] detection of grid boundaries when calculating probabilities (see [maze_2d](examples\README.md)
  example to see how the edge of the grid isn't properly synthesized)
- [x] simple backtracking
- [ ] 3D support
- [ ] generation with smaller segments
- [ ] bevy integration

As it stands, it's pretty slow to generate and often fails, espcially at bigger
sizes. 