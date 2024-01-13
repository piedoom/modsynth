use std::fmt::Display;

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub(crate) enum CellType {
    #[default]
    Air,
    Dirt,
    Grass,
    Water,
}
impl CellType {
    pub(crate) fn as_slice<'a>() -> &'a [CellType] {
        &[
            CellType::Air,
            CellType::Dirt,
            CellType::Grass,
            CellType::Water,
        ]
    }
}
impl Display for CellType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
