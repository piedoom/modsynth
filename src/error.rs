use std::fmt::Display;

/// Model synthesis error
#[derive(Debug)]
pub enum Error {
    /// Could not be solved
    Impossible,
    /// Tried to flatten a cell that wasn't solved first
    UnsolvedFlatten,
    /// Requested `Id` doesn't exist
    NotExist,
    /// Not enough info to solve
    MissingInformation,
    /// Attempted to solve something that was already solved completely
    AlreadySolved,
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Error::Impossible => "Impossible to solve with given constraints",
                Error::UnsolvedFlatten => "Attempted to flatten an unsolved cell",
                Error::NotExist => "Attempted to take an operation on a cell that does not exist",
                Error::MissingInformation => "Could not solve further without more information",
                Error::AlreadySolved => "Attepted to solve a cell that is already solved",
            }
        )
    }
}

impl std::error::Error for Error {}
