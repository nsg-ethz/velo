//! Module that samples and analyzes traffic matrices

mod cluster;
mod clustering;
mod sampler;

pub use cluster::*;
pub use clustering::*;
pub use sampler::*;

use crate::performance::TrafficClass;

use std::collections::HashMap;

/// The type of a traffic matrix
pub type TrafficMatrix<P> = HashMap<TrafficClass<P>, f64>;
