use std::collections::{hash_map::Entry, BTreeMap, BTreeSet, HashMap};

use bgpsim::prelude::*;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator};
use itertools::Itertools;
use rayon::prelude::*;

use super::{NList, PrefixSettings, TUncertainty, Uncertainty, Velo};
use crate::{
    algorithms::{EdgeId, GraphList, NodeId},
    performance::TrafficClass,
    traffic_matrix::TrafficMatrix,
};

/// Extract the set of equivalence classes for all prefixes.
pub(super) fn find<'a, P: Prefix + Send + Sync + 'static>(
    config: &'a Velo<P>,
    current_state: &'a HashMap<P, Vec<RouterId>>,
    traffic_matrix: &'a TrafficMatrix<P>,
) -> BTreeMap<PrefixSettings<'a>, Vec<(P, NList<f64>)>> {
    let mut tm = vectorize(config, traffic_matrix, current_state.len());

    let mut pecs = BTreeMap::<PrefixSettings, Vec<(P, NList<f64>)>>::new();

    // first, go through those that are special
    for ((prefixes, _), uncertainty) in config.uncertainty.iter().zip(&config.uncertainty_data) {
        for p in prefixes {
            let current_state = &config.current_state_data[p];
            let te_paths = config.te_path_data.get(p);
            let Some(x) = tm.remove(p) else { continue };
            let settings = PrefixSettings {
                te_paths,
                current_state,
                uncertainty,
            };
            pecs.entry(settings).or_default().push((*p, x));
        }
    }

    let default_uncertainty = config.uncertainty_data.last().unwrap();

    for (p, x) in tm {
        let current_state = &config.current_state_data[&p];
        let te_paths = config.te_path_data.get(&p);
        let settings = PrefixSettings {
            te_paths,
            current_state,
            uncertainty: default_uncertainty,
        };
        pecs.entry(settings).or_default().push((p, x));
    }

    pecs
}

/// Vectorize the traffic matrix into mapping from prefix to its row.
fn vectorize<P: Prefix + Send + Sync + 'static>(
    config: &Velo<P>,
    tm: &TrafficMatrix<P>,
    n: usize,
) -> HashMap<P, NList<f64>> {
    let pb = config.progress_bar("Preparing traffic matrix", tm.len() / 10_000, false);

    if !config.cluster_parallel {
        return _vectorize(config, tm, n, &pb);
    }

    let chunk_size = ((tm.len() * 2 - 1) / rayon::current_num_threads()).max(1);
    let tm = tm.iter().collect_vec();

    let chunk_iter = tm.par_chunks(chunk_size);
    let vectorized = ParallelIterator::map_with(chunk_iter, pb, |pb, tm| {
        _vectorize(config, tm.iter().copied(), n, pb)
    });
    ParallelIterator::reduce(
        vectorized,
        HashMap::<P, NList<f64>>::default,
        |mut acc, elem| {
            for (p, x) in elem {
                match acc.entry(p) {
                    Entry::Vacant(e) => {
                        e.insert(x);
                    }
                    Entry::Occupied(mut e) => e.get_mut().zip_mut(&x, |a, b| *a += *b),
                }
            }
            acc
        },
    )
}

fn _vectorize<'a, I, P: Prefix>(
    config: &'a Velo<P>,
    tm: I,
    n: usize,
    pb: &ProgressBar,
) -> HashMap<P, NList<f64>>
where
    I: IntoIterator<Item = (&'a TrafficClass<P>, &'a f64)>,
{
    let topo = &config.topo;
    let mut data: HashMap<P, NList<f64>> = HashMap::with_capacity(n);

    for (i, (&TrafficClass { src, dst }, x)) in tm.into_iter().enumerate() {
        let src = topo.topo_id(src);
        data.entry(dst).or_insert_with(|| NList::new(topo))[src] += x;

        if i % 10_000 == 0 {
            pb.inc(1);
        }
    }

    data
}

/// Prepare the uncertainty data. `config.uncertainty_data` will have one more element than
/// `config.uncertainty`: the last element is the transformed default uncertainty.
pub(super) fn prepare_uncertainty<P: Prefix + Send + Sync + 'static>(config: &mut Velo<P>) {
    config.uncertainty_data = config
        .uncertainty
        .iter()
        .map(|(_, uncertainty)| transform_uncertainty(uncertainty, config))
        .collect_vec();

    // insert the default uncertainty last
    config
        .uncertainty_data
        .push(transform_uncertainty(&config.default_uncertainty, config));
}

/// Fills `config.current_state_data` with the prepared current state data
pub(super) fn prepare_current_state<'a, P: Prefix + Send + Sync + 'static>(
    config: &'a mut Velo<P>,
    current_state: &HashMap<P, Vec<RouterId>>,
) {
    let pb = config.progress_bar("Preparing the current state", current_state.len(), false);
    if config.cluster_parallel {
        config.current_state_data = current_state
            .par_iter()
            .progress_with(pb)
            .map(|(p, state)| (*p, transform_state(state, config)))
            .collect();
    } else {
        config.current_state_data = current_state
            .iter()
            .progress_with(pb)
            .map(|(p, state)| (*p, transform_state(state, config)))
            .collect();
    }
}

/// Fills `config.te_path_data with the transformed traffic engineering paths.
pub(super) fn prepare_te_paths<'a, P: Prefix>(config: &'a mut Velo<P>) {
    config.te_path_data = config
        .te_paths
        .iter()
        .map(|(p, paths)| (*p, transform_te_paths(paths, config)))
        .collect();
}

pub(super) fn transform_te_paths<P>(
    from: &HashMap<(RouterId, RouterId), Vec<Vec<RouterId>>>,
    config: &Velo<P>,
) -> BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<EdgeId>>>> {
    let mut te_paths: BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<EdgeId>>>> = Default::default();
    let r = |x: &RouterId| config.topo.topo_id(config.external_neighbors.get(x).unwrap_or(x));
    for ((src, dst), paths) in from {
        te_paths.entry(r(src)).or_default().insert(r(dst), config.topo.paths(paths));
    }
    te_paths
}

fn transform_state<P>(from: &[RouterId], config: &Velo<P>) -> BTreeSet<NodeId> {
    from.iter()
        .map(|x| config.topo.topo_id(config.external_neighbors.get(x).unwrap_or(x)))
        .collect()
}

/// Transform a Uncertainty into the topology space
fn transform_uncertainty<P>(from: &Uncertainty, config: &Velo<P>) -> TUncertainty {
    let t_eg = |x: &BTreeMap<RouterId, BTreeSet<u32>>| {
        x.iter()
            .map(|(r, lps)| {
                (
                    config.topo.topo_id(config.external_neighbors[r]),
                    lps.clone(),
                )
            })
            .collect::<BTreeMap<NodeId, BTreeSet<u32>>>()
    };
    let t_in = |x: &BTreeSet<RouterId>| {
        x.iter().map(|x| config.topo.topo_id(x)).collect::<BTreeSet<NodeId>>()
    };

    match from {
        Uncertainty::Egress { egresses } => TUncertainty::Egress {
            egresses: t_eg(egresses),
        },
        Uncertainty::Ingress { ingresses } => TUncertainty::Ingress {
            ingresses: t_in(ingresses),
        },
        Uncertainty::Both {
            egresses,
            ingresses,
        } => TUncertainty::Both {
            egresses: t_eg(egresses),
            ingresses: t_in(ingresses),
        },
    }
}
