//! Implementation of the autodiff primitives: forward-mode dual numbers, the
//! reverse-mode tape, and the shared `Scalarish` trait.

mod dual;
mod scalarish;
mod tape;

pub use dual::Dual;
pub use scalarish::Scalarish;
pub use tape::{Tape, TapeNode, Var};
