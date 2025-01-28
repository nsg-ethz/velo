//! Algorithms for finding the worst-case link loads

use std::{
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap},
    iter::FusedIterator,
    rc::Rc,
};

use bgpsim::prelude::*;
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use maplit::btreeset;
use ordered_float::NotNan;
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec};

use crate::{
    algorithms::{
        EdgeId, EdgeList, GraphList, NodeId, NodeList, ShortestPathDag, Topology, TopologyType,
    },
    MyProgressIterator,
};

use super::{
    EList, Failures, NList, PerformanceReport, PrefixData, VeloAnalysis, WorstCasePrefixState,
    WorstCaseState,
};
type Dag = ShortestPathDag<NotNan<f64>, TopologyType>;
type EFailures = SmallVec<[EdgeId; 4]>;
type WCEgresses = SmallVec<[NodeId; 2]>;
type WCIngress = NodeId;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) enum WCState {
    Egresses(WCEgresses),
    Ingress(WCIngress),
    Both {
        ingress: NodeId,
        egress: NodeId,
    },
    #[default]
    Unchanged,
}

impl From<WCEgresses> for WCState {
    fn from(value: WCEgresses) -> Self {
        Self::Egresses(value)
    }
}

impl From<WCIngress> for WCState {
    fn from(value: WCIngress) -> Self {
        Self::Ingress(value)
    }
}

impl WCState {
    pub(super) fn transform(self, topo: &Topology) -> Option<WorstCasePrefixState> {
        match self {
            WCState::Egresses(egresses) => Some(WorstCasePrefixState::Egresses(
                egresses.into_iter().map(|r| topo.net_id(r)).collect(),
            )),
            WCState::Ingress(ingress) => Some(WorstCasePrefixState::Ingress(topo.net_id(ingress))),
            WCState::Both { ingress, egress } => Some(WorstCasePrefixState::Both {
                ingress: topo.net_id(ingress),
                egress: topo.net_id(egress),
            }),
            WCState::Unchanged => None,
        }
    }
}

impl<'a, P: Prefix + Sync + Send + 'static> VeloAnalysis<'a, P> {
    /// Create the final performance report
    fn create_report(
        &self,
        loads: EList<f64>,
        states: EList<(Option<HashMap<P, WCState>>, EFailures)>,
    ) -> PerformanceReport<P> {
        let transform_failures = |v: EFailures| {
            v.into_iter().map(|e| self.config.topo.net_link(e)).collect::<Failures>()
        };
        let transform_state = |state: HashMap<P, WCState>| {
            state
                .into_iter()
                .map(|(p, s)| (p, s.transform(&self.config.topo)))
                .filter_map(|(p, s)| s.map(|s| (p, s)))
                .collect::<HashMap<P, _>>()
        };

        let loads = loads
            .into_idx_val()
            .map(|(e, load)| (self.config.topo.net_link(e), load))
            .collect();

        let states = states
            .into_idx_val()
            .map(|(e, (inp, fail))| {
                (
                    self.config.topo.net_link(e),
                    WorstCaseState {
                        routing_inputs: inp.map(transform_state),
                        failures: transform_failures(fail),
                    },
                )
            })
            .collect();

        PerformanceReport {
            pos_bounds: self.error_pos,
            neg_bounds: self.error_neg,
            loads,
            states,
        }
    }

    /// This is the main function to perform the complete analysis on all failure scenarios. This
    /// function takes three arguments:
    /// - `k`: Number of prefixes that can change (set to `None` if all routing inputs can change),
    /// - `l`: Number of simultaneous link falures
    /// - `keep_state`: Whether to return the worst-case state.
    pub fn analyze(&self, k: Option<usize>, l: usize, keep_state: bool) -> PerformanceReport<P> {
        if k == Some(0) {
            return self.analyze_fixed_state(l);
        }
        if l == 0 {
            let (loads, states) = self.analyze_in_failure(&[], k, keep_state, true);
            let states = states.map(|_, s| (s, EFailures::new()));
            return self.create_report(loads, states);
        }

        let mut loads: EList<f64> = EList::new(&self.config.topo);
        let mut states: EList<_> = EList::new(&self.config.topo);

        let scenarios = if self.config.directional_link_failures {
            self.config.topo.edge_combinations(l)
        } else {
            self.config.topo.link_combinations(l)
        };
        let num_cases = scenarios.len();
        let results = scenarios
            .into_par_iter()
            .progress_with(self.config.progress_bar(
                format!(
                    "Find worst-case (p={}, k={}, l={l})",
                    self.data.len(),
                    k.map(|x| x.to_string()).unwrap_or("∞ ".to_string()),
                ),
                num_cases,
                true,
            ))
            .map(|failures: Vec<EdgeId>| EFailures::from(failures))
            .map(|failures: EFailures| {
                (
                    self.analyze_in_failure(&failures, k, keep_state, false),
                    failures,
                )
            })
            .collect::<Vec<_>>();

        for ((s_loads, s_states), failures) in results {
            // update the maximum values
            loads
                .iter_mut()
                .zip(states.iter_mut())
                .zip(s_loads.into_iter().zip(s_states.into_iter()))
                .for_each(|((cur, cur_state), (new, new_state))| {
                    if *cur < new {
                        *cur = new;
                        *cur_state = (new_state, failures.clone());
                    }
                })
        }

        self.create_report(loads, states)
    }

    fn analyze_fixed_state(&self, l: usize) -> PerformanceReport<P> {
        let mut loads: EList<f64> = EList::new(&self.config.topo);
        let mut states: EList<_> = EList::new(&self.config.topo);

        let scenarios = if self.config.directional_link_failures {
            self.config.topo.edge_combinations(l)
        } else {
            self.config.topo.link_combinations(l)
        };
        let num_cases = scenarios.len();

        let pb = self.config.progress_bar(
            format!("Find worst-case (p={}, k=0, l={l})", self.data.len()),
            num_cases,
            true,
        );

        let results = scenarios
            .into_par_iter()
            .progress_with(pb)
            .map(|failures: Vec<EdgeId>| EFailures::from(failures))
            .map(|failures: EFailures| {
                let mut dags = self.compute_dags(&failures);
                let mut loads = EList::new(&self.config.topo);

                // go through each prefix for egress computation
                for data in self.data.iter() {
                    let current = data.get_current_state(&self.config.topo, &mut dags, false);
                    loads.zip_mut(&current, |cur, new| *cur += *new);
                }

                (loads, failures)
            })
            .collect::<Vec<_>>();

        for (s_loads, failures) in results {
            // update the maximum values
            loads.iter_mut().zip(states.iter_mut()).zip(s_loads.into_iter()).for_each(
                |((cur, cur_state), new)| {
                    if *cur < new {
                        *cur = new;
                        *cur_state = (Default::default(), failures.clone());
                    }
                },
            )
        }

        self.create_report(loads, states)
    }

    /// Perform the analysis in a specific failure scenario, ignoring the current state.
    ///
    /// In this function, we assume that all prefixes can change arbitrarily.
    fn analyze_worst_case_in_failure(
        &self,
        failures: &[EdgeId],
        keep_state: bool,
        progress: bool,
    ) -> (EList<f64>, EList<Option<HashMap<P, WCState>>>) {
        // prepare the dags.
        let mut dags = self.compute_dags(failures);

        let mut loads: EList<f64> = EList::new(&self.config.topo);
        let mut states: EList<Option<HashMap<P, WCState>>> = EList::new(&self.config.topo);

        // do the iteration
        let iter = if progress {
            self.data.iter().my_progress_config(
                format!("Find worst-case (p={}, k=∞ , l=0)", self.data.len()),
                true,
                self.config,
            )
        } else {
            self.data.iter().my_progress("", true, false)
        };

        // go through each prefix
        for data in iter {
            let worst_case = data.get_worst_case_state(&self.config.topo, &mut dags, false);

            loads.zip_mut(&worst_case, |cur, (new, _)| *cur += *new);
            if keep_state {
                states.zip_mut_move(worst_case, |cur, (_, new)| {
                    if cur.is_none() {
                        *cur = Some(HashMap::new());
                    }
                    for (prefix, _) in data.prefixes.iter().copied() {
                        cur.as_mut().unwrap().insert(prefix, new.clone());
                    }
                })
            }
        }

        (loads, states)
    }

    /// Perform the analysis in a specific failure scenario
    fn analyze_in_failure(
        &self,
        failures: &[EdgeId],
        k: Option<usize>,
        keep_state: bool,
        progress: bool,
    ) -> (EList<f64>, EList<Option<HashMap<P, WCState>>>) {
        let Some(num_input_changes) = k else {
            // Find the absolute worst-case, allowing all routing inputs to change
            return self.analyze_worst_case_in_failure(failures, keep_state, progress);
        };

        let mut dags = self.compute_dags(failures);

        // prepare the result
        let mut ir: EList<IntermediateResult<P>> = EList::from_fn(&self.config.topo, |_| {
            IntermediateResult::new(num_input_changes)
        });

        let iter = if progress {
            self.data.iter().my_progress_config(
                format!(
                    "Find worst-case (p={}, k={num_input_changes}, l=0)",
                    self.data.len(),
                ),
                true,
                self.config,
            )
        } else {
            self.data.iter().my_progress("", true, false)
        };

        // go through each prefix
        for data in iter {
            // get the worst-case demand based on the cluster
            let worst_case = data.get_worst_case_state(&self.config.topo, &mut dags, true);

            // get the current demand
            let current = data.get_current_state(&self.config.topo, &mut dags, true);

            // fill the intermediate result
            ir.iter_mut().zip(worst_case.into_iter().zip(current)).for_each(
                |(ir, ((worst_case, worst_state), current))| {
                    // push each prefix, scaled down by its factor
                    ir.push_many(
                        worst_case,
                        worst_state,
                        current,
                        &data.prefixes,
                        data.total_demand,
                    );
                },
            )
        }

        // finish the intermediate result by summing all elements in the heap, while keeping all
        // others in the current state.
        let result = ir.map(|_, ir| ir.finish(keep_state));

        // split the data
        let loads = EList::from_other(&result, |_, (l, _)| *l);
        let result = result.map(|_, (_, x)| x);

        (loads, result)
    }

    /// Compute the DAG for each border router
    pub(super) fn compute_dags(&self, failure_list: &[EdgeId]) -> DagCache {
        let mut failures = EdgeList::new(&self.config.topo);
        for e in failure_list {
            failures[e] = true;
        }

        DagCache::new(
            self.config
                .egress_routers
                .iter()
                .map(|b| self.config.topo.topo_id(b))
                .map(|b| {
                    (
                        b,
                        ShortestPathDag::new_with_failures(&self.config.topo, b, &failures)
                            .map(|x| NotNan::new(x as f64).unwrap()),
                    )
                })
                .collect(),
        )
    }
}

impl<'a, P> PrefixData<'a, P> {
    /// Compute the link load in the given state, with the given ingresses and egresses.
    fn get_link_loads(
        &self,
        ingresses: &NList<f64>,
        egresses: &BTreeSet<NodeId>,
        topo: &Topology,
        dags: &mut DagCache,
    ) -> EList<f64> {
        let dag = dags.get(egresses);

        let mut link_loads = EList::new(topo);
        let mut dag_loads = ingresses.clone();

        // first, apply all paths
        for (egress, ingress_paths) in self.get_te_paths() {
            for (ingress, paths) in ingress_paths {
                if dag.egress(ingress) != Some(*egress) {
                    // path is not used!
                    continue;
                }

                let demand = dag_loads[ingress];
                let per_path_demand = demand / paths.len() as f64;
                dag_loads[ingress] = 0.0;

                // increase the demand
                paths.iter().flatten().for_each(|e| link_loads[e] += per_path_demand);
            }
        }

        // then, go through all remaining demands
        for r in dag.toposort() {
            let edges = &dag[r].next;
            let load = dag_loads[r] / edges.len() as f64;

            for (n, e) in edges {
                dag_loads[n] += load;
                link_loads[e] += load;
            }
        }

        link_loads
    }

    /// Compute the current state of the prefix data.
    ///
    /// - If `normalized` is set to `true`, then this function will return the maximum link loads
    ///   for the normalized traffic. To get the actual load, you will need to multiply the results
    ///   by `self.total_demand` (or by the total demand for a specific prefix in `self`).
    /// - If `normalized` is set to `false`, then this function simply returns the link load in the
    ///   current state, that would be generated by all prefixes in `self` simultaneously.
    pub(super) fn get_current_state(
        &self,
        topo: &Topology,
        dags: &mut DagCache,
        normalized: bool,
    ) -> EList<f64> {
        let mut load =
            self.get_link_loads(&self.current_norm_demand, self.current_state, topo, dags);
        // undo the normalization if `normalized` is set to false
        if !normalized {
            load.iter_mut().for_each(|x| *x *= self.total_demand);
        }
        load
    }

    /// Compute the worst-case link load for the given prefix.
    ///
    /// - If `normalized` is set to `true`, then this function will return the maximum link loads
    ///   for the normalized traffic. To get the actual load, you will need to multiply the results
    ///   by `self.total_demand` (or by the total demand for a specific prefix in `self`).
    /// - If `normalized` is set to `false`, then this function simply returns the link load in the
    ///   current state, that would be generated by all prefixes in `self` simultaneously.
    #[inline(always)]
    fn get_worst_case_state(
        &self,
        topo: &Topology,
        dags: &mut DagCache,
        normalized: bool,
    ) -> EList<(f64, WCState)> {
        let mut result = match self.uncertainty {
            super::TUncertainty::Egress { egresses } => {
                get_wcs_only_egress(self, egresses, topo, dags)
            }
            super::TUncertainty::Ingress { ingresses } => {
                get_wcs_only_ingress(self, ingresses, topo, dags)
            }
            super::TUncertainty::Both {
                egresses,
                ingresses,
            } => get_wcs_both_ingresses_and_egresses(self, ingresses, egresses, topo, dags),
        };
        // undo the normalization if `normalized` is set to false
        if !normalized {
            result.iter_mut().for_each(|(x, _)| *x *= self.total_demand);
        }
        result
    }
}

/// Get the worst-case state if only ingresses can change.
///
/// This function returns the loads in terms of the normalized link loads, i.e., it will return
/// 1.0 if all traffic for all prefixes in `data` *can* be routed over the same link.
fn get_wcs_only_ingress<'a, P>(
    data: &PrefixData<'a, P>,
    ingresses: &BTreeSet<NodeId>,
    topo: &Topology,
    dags: &mut DagCache,
) -> EList<(f64, WCState)> {
    let dag = dags.get(data.current_state);

    // we traverse the DAG once, but for all ingress links together. The inner NList<f64> is the
    // respective value we maintain for each ingress node.
    let mut link_loads: EList<(f64, WCState)> = EList::new(topo);
    let mut dag_loads: NList<NList<f64>> = NList::from_fn(topo, |_| NList::new(topo));

    // start filling all possible ingress links by starting with the complete load from that router.
    ingresses.iter().for_each(|&r| dag_loads[r][r] = 1.0);

    // first, apply all paths
    for (egress, ingress_paths) in data.get_te_paths() {
        for (ingress, paths) in ingress_paths {
            if dag.egress(ingress) != Some(*egress) {
                // path is not used!
                continue;
            }

            let per_path_demand = dag_loads[ingress][ingress] / paths.len() as f64;
            dag_loads[ingress][ingress] = 0.0;

            // update that maximum link load, but only if it is larger than what we currently
            // have there
            for e in paths.iter().flatten() {
                if link_loads[e].0 < per_path_demand {
                    link_loads[e] = (per_path_demand, (*ingress).into());
                }
            }
        }
    }

    // then, go through all remaining demands
    for r in dag.toposort() {
        let edges = &dag[r].next;
        let norm = 1.0 / edges.len() as f64;
        let input_load = NodeList::from_other(&dag_loads[r], |_, l| *l * norm);

        for (n, e) in edges {
            dag_loads[n].zip_mut(&input_load, |at_n, from_r| *at_n += *from_r);
            let (worst_ingress, worst_load) =
                input_load.idx_iter().max_by(|&(_, a), &(_, b)| a.total_cmp(b)).unwrap();
            if link_loads[e].0 < *worst_load {
                link_loads[e] = (*worst_load, worst_ingress.into());
            }
        }
    }

    link_loads
}

/// Get the worst-case state if only egresses can change.
///
/// This function returns the loads in terms of the normalized current state.
#[inline(always)]
fn get_wcs_only_egress<'a, P>(
    data: &PrefixData<'a, P>,
    egresses: &BTreeMap<NodeId, BTreeSet<u32>>,
    topo: &Topology,
    dags: &mut DagCache,
) -> EList<(f64, WCState)> {
    if let Some(te_paths) = data.te_paths {
        get_wcs_only_egress_with_te_paths(data, egresses, te_paths, topo, dags)
    } else {
        get_wcs_only_egress_no_te_paths(data, egresses, topo, dags)
    }
}

/// Get the worst-case state if only egresses can change, and there are no TE paths.
///
/// This function returns the loads in terms of the normalized current state.
fn get_wcs_only_egress_no_te_paths<'a, P>(
    data: &PrefixData<'a, P>,
    egresses: &BTreeMap<NodeId, BTreeSet<u32>>,
    topo: &Topology,
    dags: &mut DagCache,
) -> EList<(f64, WCState)> {
    // for each prefix, we find the worst-case total load for each link
    let mut link_loads: EList<(f64, WCState)> = EList::new(topo);

    // try to find the worst-case routing inputs for prefix p
    for (egress, _) in egresses.iter() {
        let egress = *egress;
        let dag = dags.get(&btreeset![egress]);

        // prepare all demands, i.e., the traffic that flows through each node in that DAG.
        let mut dag_loads: NList<f64> = data.current_norm_demand.clone();
        // iterate over all nodes in topological order
        for r in dag.toposort() {
            let edges = &dag[r].next;
            let load = dag_loads[r] / edges.len() as f64;
            for (n, e) in edges {
                dag_loads[n] += load;
                let (cur_load, cur_egress) = &mut link_loads[e];
                if *cur_load < load {
                    *cur_load = load;
                    *cur_egress = smallvec![egress].into();
                }
            }
        }
    }

    link_loads
}

/// Get the worst-case state if only egresses can change, and there are TE paths configured.
///
/// This function returns the loads in terms of the normalized current state.
fn get_wcs_only_egress_with_te_paths<'a, P>(
    data: &PrefixData<'a, P>,
    egresses: &BTreeMap<NodeId, BTreeSet<u32>>,
    te_paths: &BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<EdgeId>>>>,
    topo: &Topology,
    dags: &mut DagCache,
) -> EList<(f64, WCState)> {
    // for each prefix, we find the worst-case total load for each link
    let mut link_loads: EList<(f64, WCState)> = EList::new(topo);

    // first, get the set of all special egresses that we need to consider, and construct the
    // matching egress iterator
    let te_egresses = egresses
        .iter()
        .filter(|(e, _)| te_paths.contains_key(e))
        .map(|(e, lps)| (*e, lps))
        .collect();
    let te_egress_iter = MatchingEgressIterator::new(te_egresses);

    // construct the set of other egresses, that are not already covered using te_egresses
    let mut other_egresses = egresses
        .iter()
        .filter(|(e, _)| !te_paths.contains_key(e))
        .map(|(e, lps)| Some((*e, lps)))
        .collect_vec();
    // also push the case where no other egress is available
    other_egresses.push(None);

    // iterate over all combinations of TE egresses
    for (te_egresses, current_lps) in te_egress_iter {
        // now, we need to iterate over all other egresses (but not all combinations, because
        // those are guaranteed to be free of deflection!)
        'other_egresses: for other in &other_egresses {
            let mut te_egresses = te_egresses.clone();

            // `other` may be empty. If not, then check if this is possible with current_lps
            if let Some((other_egress, other_lps)) = other {
                // if current_lps is None, then `other_egress` is always an option
                let current_lps = current_lps.as_ref().map(|x| x.as_ref()).unwrap_or(other_lps);
                // check if their intersection is empty
                if current_lps.is_disjoint(other_lps) {
                    // they are incompatible
                    continue 'other_egresses;
                }
                // if they are not disjoint, then they are compatible.
                te_egresses.insert(*other_egress);
            }

            // if te_egresses is now empty, then we don't need to do anything here.
            if te_egresses.is_empty() {
                continue 'other_egresses;
            }

            // if we reach this far, the variable `te_egresses` contains a set of egresses, and
            // we now want to check what is the resulting link load in that state.
            let loads = data.get_link_loads(&data.current_norm_demand, &te_egresses, topo, dags);

            // finally, update the link loads if it is better
            let state: WCState = te_egresses.iter().copied().collect::<WCEgresses>().into();

            for (e, load) in loads.into_idx_val() {
                let (cur_load, cur_egress) = &mut link_loads[e];
                if *cur_load < load {
                    *cur_load = load;
                    *cur_egress = state.clone();
                }
            }
        }
    }

    link_loads
}

/// Get the worst-case state if only egresses can change.
///
/// Here, we apply the following insight:
/// - No matter the egress state, the worst-case ingress distribution is when all traffic enters
///   from one ingress point.
/// - No matter the ingress state, the worst-case egress points are when all traffic leaves at one
///   single egress.
/// - Even if traffic engineering is used, since we only use a single ingress, we can safely
///   consider a single egress.
///
/// The algorithm looks like a merging of the `get_wcs_only_ingress` and `get_wcs_only_egress`.
///
/// This function returns the loads in terms of the normalized link loads, i.e., it will return
/// 1.0 if all traffic for all prefixes in `data` *can* be routed over the same link.
fn get_wcs_both_ingresses_and_egresses<'a, P>(
    data: &PrefixData<'a, P>,
    ingresses: &BTreeSet<NodeId>,
    egresses: &BTreeMap<NodeId, BTreeSet<u32>>,
    topo: &Topology,
    dags: &mut DagCache,
) -> EList<(f64, WCState)> {
    // for each prefix, we find the worst-case total load for each link
    let mut link_loads: EList<(f64, WCState)> = EList::new(topo);

    // try to find the worst-case routing inputs for prefix p
    for (&egress, _) in egresses.iter() {
        let dag = dags.get(data.current_state);

        // we traverse the DAG once, but for all ingress links together. The inner NList<f64> is the
        // respective value we maintain for each ingress node.
        let mut dag_loads: NList<NList<f64>> = NList::from_fn(topo, |_| NList::new(topo));

        // start filling all possible ingress links by adding their dag loads to 1.
        ingresses.iter().for_each(|&r| dag_loads[r][r] = 1.0);

        // first, apply all paths that are going to that egress.
        for (&ingress, paths) in data.get_te_paths().get(&egress).into_iter().flatten() {
            if dag.egress(ingress) != Some(egress) {
                // path is not used!
                continue;
            }

            let per_path_demand = dag_loads[ingress][ingress] / paths.len() as f64;
            dag_loads[ingress][ingress] = 0.0;

            // update that maximum link load, but only if it is larger than what we currently
            // have there
            for e in paths.iter().flatten() {
                if link_loads[e].0 < per_path_demand {
                    link_loads[e] = (per_path_demand, WCState::Both { ingress, egress })
                }
            }
        }

        // then, go through all remaining demands
        for r in dag.toposort() {
            let edges = &dag[r].next;
            let norm = 1.0 / edges.len() as f64;
            let input_load = NodeList::from_other(&dag_loads[r], |_, l| *l * norm);

            for (n, e) in edges {
                dag_loads[n].zip_mut(&input_load, |at_n, from_r| *at_n += *from_r);
                let (worst_ingress, worst_load) =
                    input_load.idx_iter().max_by(|&(_, a), &(_, b)| a.total_cmp(b)).unwrap();
                if link_loads[e].0 < *worst_load {
                    link_loads[e] = (
                        *worst_load,
                        WCState::Both {
                            ingress: worst_ingress,
                            egress,
                        },
                    );
                }
            }
        }
    }

    link_loads
}

struct MatchingEgressIterator<'a> {
    egresses: Vec<(NodeId, &'a BTreeSet<u32>)>,
    current_state: Vec<(bool, Option<Rc<BTreeSet<u32>>>)>,
    first_iter: bool,
}

impl<'a> MatchingEgressIterator<'a> {
    fn new(egresses: Vec<(NodeId, &'a BTreeSet<u32>)>) -> Self {
        let current_state = vec![(false, None); egresses.len()];
        Self {
            egresses,
            current_state,
            first_iter: true,
        }
    }
}

impl<'a> Iterator for MatchingEgressIterator<'a> {
    type Item = (BTreeSet<NodeId>, Option<Rc<BTreeSet<u32>>>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.first_iter {
            self.first_iter = false;
            return Some((Default::default(), None));
        }

        debug_assert_eq!(self.current_state.len(), self.egresses.len());
        // the main loop that is repeated until we find a valid solution
        'main: loop {
            // go back until we reach an element that stores `false`.
            'backtrack: loop {
                // otherwhise, we can continue
                let Some((last_elem, _)) = self.current_state.pop() else {
                    // break condition
                    return None;
                };
                // if last_elem is true, then we need to go back even further.
                if last_elem {
                    continue 'backtrack;
                }
                // in that branch, it is true. we break out of the loop, and can dive in by setting the
                // next one to `true`.
                break 'backtrack;
            }

            let next_idx = self.current_state.len();
            let (_, next_egress_lps) = &self.egresses[next_idx];

            // get the last set of local-pref values
            let lps = self
                .current_state
                .last()
                .and_then(|(_, lps)| lps.as_ref().map(|x| x.as_ref()))
                .unwrap_or(next_egress_lps);

            // construct the new set of possible LP values if `next_egress` is selected.
            let new_lps: BTreeSet<u32> = lps.intersection(next_egress_lps).cloned().collect();

            // if new_lps is non-empty, then that is a valid option. Otherwhise, we need to
            // backtrack one more...
            if new_lps.is_empty() {
                continue 'main;
            }

            // We can use that router!
            let new_lps = Rc::new(new_lps);
            self.current_state.push((true, Some(new_lps.clone())));

            // Set all others (that come later) to "not used" (they will be used in the next
            // iteration)
            while self.current_state.len() < self.egresses.len() {
                self.current_state.push((false, Some(new_lps.clone())));
            }

            // now, we have constructed the next iteration
            return Some((
                self.egresses
                    .iter()
                    .map(|(e, _)| *e)
                    .zip(self.current_state.iter().map(|(b, _)| *b))
                    .filter(|(_, b)| *b)
                    .map(|(e, _)| e)
                    .collect(),
                Some(new_lps.clone()),
            ));
        }
    }
}

impl<'a> FusedIterator for MatchingEgressIterator<'a> {}

#[derive(PartialEq)]
struct IntermediateElement<P> {
    prefix: P,
    state: WCState,
    diff: NotNan<f64>,
}

impl<P: Prefix> Eq for IntermediateElement<P> {}

impl<P: Prefix> PartialOrd for IntermediateElement<P> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: Prefix> Ord for IntermediateElement<P> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.diff.cmp(&self.diff)
    }
}

struct IntermediateResult<P> {
    top: BinaryHeap<IntermediateElement<P>>,
    sum_current_state: f64,
    k: usize,
}

impl<P: Prefix> IntermediateResult<P> {
    fn new(k: usize) -> Self {
        Self {
            top: BinaryHeap::with_capacity(k),
            sum_current_state: 0.0,
            k,
        }
    }
}

impl<P: Prefix> IntermediateResult<P> {
    fn push_many(
        &mut self,
        worst_case: f64,
        state: impl Into<WCState>,
        current: f64,
        prefixes: &[(P, f64)],
        total_demand: f64,
    ) {
        let state = state.into();
        let diff = NotNan::new(worst_case - current).unwrap();
        self.sum_current_state += current * total_demand;
        // for each of the prefixes, check if we need to push something in the top
        for (prefix, factor) in prefixes.iter().copied() {
            let diff = diff * factor;
            if self.top.len() < self.k {
                self.top.push(IntermediateElement {
                    prefix,
                    state: state.clone(),
                    diff,
                })
            } else if self.top.peek().unwrap().diff < diff {
                *self.top.peek_mut().unwrap() = IntermediateElement {
                    prefix,
                    state: state.clone(),
                    diff,
                }
            }
        }
    }

    fn finish(self, keep_state: bool) -> (f64, Option<HashMap<P, WCState>>) {
        let total =
            self.sum_current_state + self.top.iter().map(|x| x.diff.into_inner()).sum::<f64>();
        let mut map = None;
        if keep_state {
            map = Some(self.top.into_iter().map(|x| (x.prefix, x.state)).collect());
        }
        (total, map)
    }
}

pub(super) struct DagCache {
    dags: BTreeMap<NodeId, Dag>,
    memory: HashMap<BTreeSet<NodeId>, Dag>,
}

impl DagCache {
    fn new(dags: BTreeMap<NodeId, Dag>) -> Self {
        Self {
            dags,
            memory: Default::default(),
        }
    }

    fn get(&mut self, egresses: &BTreeSet<NodeId>) -> &Dag {
        if egresses.len() == 1 {
            &self.dags[egresses.first().unwrap()]
        } else {
            if !self.memory.contains_key(egresses) {
                self.memory.insert(
                    egresses.clone(),
                    ShortestPathDag::merge_unique(egresses.iter().map(|r| &self.dags[r]), 1.0),
                );
            }
            &self.memory[egresses]
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_matching_egress_iterator() {
        let egresses = [
            (0.into(), btreeset! {1, 2, 3}),
            (1.into(), btreeset! {2, 3}),
            (2.into(), btreeset! {3}),
        ];
        let iter = MatchingEgressIterator::new(egresses.iter().map(|(a, b)| (*a, b)).collect());

        assert_eq!(iter.count(), 8)
    }

    #[test]
    fn test_matching_egress_iterator_empty() {
        let egresses = [
            (0.into(), btreeset! {}),
            (1.into(), btreeset! {}),
            (2.into(), btreeset! {}),
        ];
        let iter = MatchingEgressIterator::new(egresses.iter().map(|(a, b)| (*a, b)).collect());

        assert_eq!(iter.count(), 1)
    }

    #[test]
    fn test_matching_egress_iterator_comb() {
        let egresses = [
            (0.into(), btreeset! {1, 3}),
            (1.into(), btreeset! {1, 2}),
            (2.into(), btreeset! {2, 3}),
        ];
        let iter = MatchingEgressIterator::new(egresses.iter().map(|(a, b)| (*a, b)).collect());

        assert_eq!(iter.count(), 7)
    }
}
