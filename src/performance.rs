//! Data types and utility functions to analyze performance metrics on a network.

use std::collections::{hash_map::Entry, HashMap};

use bgpsim::{
    forwarding_state::ForwardingState,
    ospf::OspfImpl,
    prelude::{Network, NetworkFormatter},
    types::{Prefix, RouterId},
};
use serde::{Deserialize, Serialize};

/// A traffic class describes the traffic from an origin to a destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TrafficClass<P> {
    /// Where does the traffic start.
    pub src: RouterId,
    /// What is the destination towards the traffic.
    pub dst: P,
}

impl<'n, P: Prefix, Q, Ospf: OspfImpl> NetworkFormatter<'n, P, Q, Ospf> for TrafficClass<P> {
    fn fmt(&self, net: &'n Network<P, Q, Ospf>) -> String {
        format!("({}:{})", self.src.fmt(net), self.dst)
    }
}

/// Compute the load (as a fraction of the input rate) on each link, assuming that the traffic is
/// split equally.
///
/// **Warning**: This function assumes the absence of forwarding loops!
pub fn ecmp_link_load<P: Prefix>(
    fw_state: &ForwardingState<P>,
    tc: TrafficClass<P>,
) -> HashMap<(RouterId, RouterId), f64> {
    let mut result = HashMap::new();
    _ecmp_link_load_recursive(fw_state, tc.src, tc.dst, 1.0, &mut result);
    result
}

fn _ecmp_link_load_recursive<P: Prefix>(
    fw_state: &ForwardingState<P>,
    src: RouterId,
    dst: P,
    cur: f64,
    result: &mut HashMap<(RouterId, RouterId), f64>,
) {
    let nhs = fw_state.get_next_hops(src, dst);
    let out_fraction = cur / nhs.len() as f64;
    for nh in nhs {
        match result.entry((src, *nh)) {
            Entry::Vacant(e) => {
                e.insert(out_fraction);
            }
            Entry::Occupied(mut e) => *e.get_mut() += out_fraction,
        }
        _ecmp_link_load_recursive(fw_state, *nh, dst, out_fraction, result);
    }
}

/// Compute the total load on each link according to the given traffic matrix.
pub fn total_link_load<P: Prefix>(
    fw_state: &ForwardingState<P>,
    traffic_matrix: &HashMap<TrafficClass<P>, f64>,
) -> HashMap<(RouterId, RouterId), f64> {
    let mut result: HashMap<(RouterId, RouterId), f64> = HashMap::new();
    for (tc, demand) in traffic_matrix {
        let link_load = ecmp_link_load(fw_state, *tc);
        for (link, fraction) in link_load {
            let load = fraction * demand;
            match result.entry(link) {
                Entry::Vacant(e) => {
                    e.insert(load);
                }
                Entry::Occupied(mut e) => *e.get_mut() += load,
            }
        }
    }
    result
}

/// Check wether the traffic matrix would cause links to be overloaded given the current forwarding
/// state.
///
/// If a link is not present in `link_capacity`, that link is assumed to have infinite capacity.
pub fn check_link_overload<P: Prefix>(
    fw_state: &ForwardingState<P>,
    traffic_matrix: &HashMap<TrafficClass<P>, f64>,
    link_capacity: &HashMap<(RouterId, RouterId), f64>,
) -> OverloadResult {
    let link_load = total_link_load(fw_state, traffic_matrix);
    let mut overloaded = Vec::new();
    for (link, cap) in link_capacity {
        let load = link_load.get(link).copied().unwrap_or(0.0);
        if load >= *cap {
            overloaded.push(*link);
        }
    }
    if overloaded.is_empty() {
        OverloadResult::Satisfied
    } else {
        OverloadResult::LinkOverload(overloaded)
    }
}

/// The outcome of checking the network for a link overload.
#[derive(Debug, Clone, PartialEq, Eq, Hash, NetworkFormatter)]
pub enum OverloadResult {
    /// No link in the network is overloaded
    Satisfied,
    /// Some links in the network are overloaded.
    LinkOverload(Vec<(RouterId, RouterId)>),
}
