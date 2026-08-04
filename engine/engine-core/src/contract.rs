//! Descriptor checks and authoritative runtime validation.

mod descriptors;
mod runtime;

pub(crate) use descriptors::validate_descriptors;
#[cfg(test)]
pub(crate) use descriptors::{validate_action_space, validate_encoding};
#[cfg(test)]
pub(crate) use runtime::validate_encoded_observation;
pub(crate) use runtime::validate_erased_timestep;
#[cfg(test)]
pub(crate) use runtime::validate_typed_timestep;
