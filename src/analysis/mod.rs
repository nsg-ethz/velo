//! Module to analyze the performance of networks in different environments.

pub mod find_worst_case;
mod pecs;
pub mod state_aware_clustering;
pub mod traffic_variability;

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    hash::Hash,
};

use bgpsim::prelude::*;
use indicatif::{MultiProgress, ProgressBar};
use maplit::btreeset;
use ordered_float::NotNan;
use smallvec::SmallVec;

use crate::{
    algorithms::{EdgeId, EdgeList, GraphList, NodeId, NodeList, Topology, TopologyType},
    my_progress,
    traffic_matrix::{ClusterMode, TrafficMatrix},
};

use self::state_aware_clustering::cluster;

/// Extract the current link load of the network.
pub fn current_link_load<P: Prefix + Send + Sync + 'static, Q, Ospf: OspfImpl>(
    net: &Network<P, Q, Ospf>,
    current_state: &HashMap<P, Vec<RouterId>>,
    traffic_matrix: &TrafficMatrix<P>,
) -> HashMap<(RouterId, RouterId), f64> {
    let config = Velo::new(net);
    let velo = VeloAnalysis::new(&config, current_state, traffic_matrix);

    let mut dags = velo.compute_dags(&[]);
    let mut loads: EList<f64> = EList::new(&velo.config.topo);

    // go through each egress prefix
    for data in velo.data.iter() {
        // get the current demand
        let current = data.get_current_state(&velo.config.topo, &mut dags, false);
        loads.zip_mut(&current, |cur, new| *cur += *new);
    }

    loads
        .into_idx_val()
        .map(|(e, load)| (velo.config.topo.net_link(e), load))
        .collect()
}

/// # Verify Isp Performance
///
/// This is a builder pattern to perform end-to-end performance analysis.
#[derive(Debug)]
pub struct Velo<P> {
    /// The topology, extracted from the network.
    topo: Topology,
    /// The set of all external routers
    egress_routers: BTreeSet<RouterId>,
    /// A mapping from external to internal router
    external_neighbors: BTreeMap<RouterId, RouterId>,
    /// All router names, used for debug printing
    #[allow(dead_code)]
    router_names: HashMap<RouterId, String>,
    /// The set of traffic-engineered paths, configured on a per-prefix basis.
    te_paths: HashMap<P, HashMap<(RouterId, RouterId), Vec<Vec<RouterId>>>>,
    /// The uncertainty information to take for all prefixes if not specified in `self.uncertainty`.
    default_uncertainty: Uncertainty,
    /// The uncertainty information, as long as it is different from `self.default_uncertainty`.
    uncertainty: Vec<(BTreeSet<P>, Uncertainty)>,
    /// How should we cluster together prefixes.
    clustering: ClusterSettings,
    /// The number of preifx equivalence classes to consider. All others will be ignored.
    heaviest_pecs: Option<usize>,
    /// Perform the clustering in parallel using all available threads.
    cluster_parallel: bool,
    /// Whether to explore directional link failures, or symmetrical ones.
    directional_link_failures: bool,
    /// Whether to show a progress bar
    pub(crate) progress: bool,
    /// A muilti-progress bar if called in a pipeline of other stuff.
    pub(crate) multi_pb: Option<MultiProgress>,
    /// Place to store all TUncertainy for the VeloAnalysis
    uncertainty_data: Vec<TUncertainty>,
    /// Place to store all current states for the VeloAnalysis
    current_state_data: HashMap<P, BTreeSet<NodeId>>,
    /// Place to store all transformed te paths for VeloAnalysis
    te_path_data: HashMap<P, BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<EdgeId>>>>>,
}

/// How to perform the clustering of the data (while only clustering together prefixes with the same
/// config, i.e., those of the same equivalence class).
///
/// If set to `None`, then all prefixes will be treated individually. Otherwise, the prefixes that
/// have `Uncertainty::Egress` will be clustered according to the given settings, while all others
/// will simply be clustered according to their prefix equivalence class.
#[derive(Debug)]
pub enum ClusterSettings {
    /// Distribute clusters fairly according to the amount of traffic flowing in each equivalence
    /// class.
    FixedNum {
        /// Total number of available clusters
        num: usize,
        /// The clustering mode to use
        mode: ClusterMode,
    },
    /// Pick as many clusters as required to hit a given clustering error.
    TargetError {
        /// Target error
        target: f64,
        /// The clustering mode to use
        mode: ClusterMode,
    },
    /// Treat each prefix individually (even those that have `Uncertainty::Ingress` or
    /// `Uncertainty::Both`).
    None,
}

#[derive(Debug, PartialEq, Eq)]
/// Information on what can vary for a given prefix.
pub enum Uncertainty {
    /// The egress point of the prefix may change
    Egress {
        /// The set of possible egress routers for that prefix, including the set of local-pref
        /// values a route for that prefix may have.
        ///
        /// The `RouterId` represents an external router.
        egresses: BTreeMap<RouterId, BTreeSet<u32>>,
    },
    /// The ingress point of the prefix may change. If not, then we pick the one from the current
    /// state.
    Ingress {
        /// The set of possible ingress points for that prefix. The `RouterId` here must represent
        /// internal routers.
        ingresses: BTreeSet<RouterId>,
    },
    /// Both the ingress and the egress point of the prefix may change.
    Both {
        /// The set of possible egress routers for that prefix, including the set of local-pref
        /// values a route for that prefix may have.
        ///
        /// The `RouterId` represents an external router.
        egresses: BTreeMap<RouterId, BTreeSet<u32>>,
        /// The set of possible ingress points for that prefix. The `RouterId` here must represent
        /// internal routers.
        ingresses: BTreeSet<RouterId>,
    },
}

impl<P: Prefix + Send + Sync + 'static> Velo<P> {
    /// Create a new Velo instance.
    pub fn new<Q, Ospf: OspfImpl>(net: &Network<P, Q, Ospf>) -> Self {
        let egress_routers = net.ospf_network().external_edges().map(|e| e.int).collect();
        let external_neighbors: BTreeMap<RouterId, RouterId> =
            net.ospf_network().external_edges().map(|e| (e.ext, e.int)).collect();
        let router_names = net.devices().map(|r| (r.router_id(), r.name().to_string())).collect();
        let default_uncertainty = Uncertainty::Egress {
            egresses: external_neighbors.keys().map(|r| (*r, btreeset! {100})).collect(),
        };
        Self {
            topo: Topology::new(net),
            egress_routers,
            external_neighbors,
            router_names,
            default_uncertainty,
            uncertainty: Default::default(),
            clustering: ClusterSettings::None,
            heaviest_pecs: None,
            te_paths: Default::default(),
            cluster_parallel: true,
            directional_link_failures: false,
            progress: true,
            multi_pb: None,
            uncertainty_data: Default::default(),
            current_state_data: Default::default(),
            te_path_data: Default::default(),
        }
    }

    /// Set the default uncertainty, applied to all prefixes not overwritten manually.
    ///
    /// By default, this value is set to `Uncertainty::Egress`, where the possible egresses are all
    /// egress routers configured in the network, all of which can only advertise a route with
    /// LocalPref 100.
    ///
    /// See also `modify_default_ingresses` and `modify_default_egresses`.
    pub fn set_default_uncertainty(&mut self, uncertainty: Uncertainty) -> &mut Self {
        self.default_uncertainty = uncertainty;
        self
    }

    /// Modify the set of egresses that are used in the default uncertainty configuration.
    ///
    /// The function is called for each possible external router (including its current possible
    /// local preferences). By returning an empty set, you indicate that this external router will
    /// never advertise a prefix in the default case. Otherwise, that external router can advertise
    /// routes with one of the given LocalPref values.
    ///
    /// If you call this function with current default uncertainty set to `Uncertainty::Ingress`,
    /// then nothing will change, and `f` will not be called.
    pub fn modify_default_egresses<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(RouterId, BTreeSet<u32>) -> BTreeSet<u32>,
    {
        match &mut self.default_uncertainty {
            Uncertainty::Egress { egresses } | Uncertainty::Both { egresses, .. } => {
                self.external_neighbors.keys().for_each(|r| {
                    let old = egresses.remove(r).unwrap_or_default();
                    let new = f(*r, old);
                    if !new.is_empty() {
                        egresses.insert(*r, new);
                    }
                })
            }
            Uncertainty::Ingress { .. } => {}
        }
        self
    }

    /// Modify the set of ingresses that are used in the default uncertainty configuration.
    ///
    /// The function is called for each possible external router (including whether it is currently
    /// possible to be used). The return value tells whether this ingress should be possible
    ///
    /// If you call this function with current default uncertainty set to `Uncertainty::Egress`,
    /// then nothing will change, and `f` will not be called.
    pub fn modify_default_ingresses<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(RouterId, bool) -> bool,
    {
        match &mut self.default_uncertainty {
            Uncertainty::Ingress { ingresses } | Uncertainty::Both { ingresses, .. } => {
                self.external_neighbors.keys().for_each(|r| {
                    let old = ingresses.remove(r);
                    let new = f(*r, old);
                    if new {
                        ingresses.insert(*r);
                    }
                })
            }
            Uncertainty::Egress { .. } => {}
        }
        self
    }

    /// Get the default uncertainty, as currently configured.
    pub fn get_default_uncertainty(&self) -> &Uncertainty {
        &self.default_uncertainty
    }

    /// Configure a set of prefixes to be treated with a different uncertainty information.
    pub fn with_uncertainty<I>(&mut self, prefixes: I, uncertainty: Uncertainty) -> &mut Self
    where
        I: IntoIterator<Item = P>,
    {
        let prefixes = BTreeSet::from_iter(prefixes);

        // remove this set of prefixes from all other special cases
        for (prev_p, _) in &mut self.uncertainty {
            let intersection: Vec<P> = prefixes.intersection(prev_p).copied().collect();
            for p in intersection {
                prev_p.remove(&p);
            }
        }

        // now, go through the list and check if the same uncertainty was already configured
        // somewhere
        for (prev_p, prev_u) in &mut self.uncertainty {
            if prev_u == &uncertainty {
                prev_p.extend(prefixes);
                return self;
            }
        }

        // if we reach here, then uncertainty is truely new.
        self.uncertainty.push((prefixes, uncertainty));
        self
    }

    /// Add one exceptional traffic-engineering path to the configuration.
    pub fn add_te_path(
        &mut self,
        ingress: RouterId,
        egress: RouterId,
        prefix: P,
        path: Vec<Vec<RouterId>>,
    ) -> &mut Self {
        self.te_paths.entry(prefix).or_default().insert((ingress, egress), path);
        self
    }

    /// Use a multi-progress bar
    pub fn multi_progress(&mut self, multi_pb: MultiProgress) -> &mut Self {
        self.multi_pb = Some(multi_pb);
        self
    }

    /// Hide the progress bar when performing the analysis
    pub fn hide_progress(&mut self) -> &mut Self {
        self.progress = false;
        self
    }

    /// Configure Velo to only use the heaviest prefixes in the matrix.
    pub fn only_heaviest_prefixes(&mut self, num: usize) -> &mut Self {
        self.heaviest_pecs = Some(num);
        self
    }

    /// Cluster together prefixes that share the same configuration and current-state, and also have
    /// similar traffic patterns. The number of clusters is just an approximation, and the exact
    /// number will differ.
    pub fn with_clustering(&mut self, settings: ClusterSettings) -> &mut Self {
        self.clustering = settings;
        self
    }

    /// Perform the clustering on a single thread
    pub fn cluster_single_thread(&mut self) -> &mut Self {
        self.cluster_parallel = false;
        self
    }

    /// Set the mode to use directional link failures
    pub fn directional_link_failures(&mut self) -> &mut Self {
        self.directional_link_failures = true;
        self
    }

    /// Get a mutable reference to the underlying topology.
    ///
    /// # Safety
    /// Do not modify or add nodes!
    pub unsafe fn topo(&mut self) -> &mut Topology {
        &mut self.topo
    }

    /// Perform the analysis. For each link, this function returns both the worst-case link load,
    /// and the routing delta to the routing input that caused that input to change.
    ///
    /// The current state is a list of prefixes, and from which external routers they are
    /// advertised.
    pub fn prepare<'a>(
        &'a mut self,
        current_state: &'a HashMap<P, Vec<RouterId>>,
        traffic_matrix: &'a TrafficMatrix<P>,
    ) -> VeloAnalysis<'a, P> {
        self.uncertainty_data.clear();
        self.current_state_data.clear();
        self.te_path_data.clear();
        // prepare the current state
        pecs::prepare_current_state(self, current_state);
        pecs::prepare_uncertainty(self);
        pecs::prepare_te_paths(self);
        VeloAnalysis::new(self, current_state, traffic_matrix)
    }

    /// Prepare the set of TE paths to install them directly into `VeloAnalysis` later. Use this
    /// function only for evaluation!
    pub fn prepare_te_paths(
        &self,
        p: Vec<(RouterId, RouterId, Vec<Vec<RouterId>>)>,
    ) -> BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<EdgeId>>>> {
        let paths = p.into_iter().map(|(src, dst, paths)| ((src, dst), paths)).collect();
        pecs::transform_te_paths(&paths, self)
    }
}

impl<P> Velo<P> {
    /// Return a progress bar based on the configuration
    pub(crate) fn progress_bar(
        &self,
        msg: impl Into<String>,
        len: usize,
        keep: bool,
    ) -> ProgressBar {
        let pb = my_progress(msg, len, keep, self.progress);
        if let Some(multi_pb) = &self.multi_pb {
            multi_pb.insert_from_back(1, pb)
        } else {
            pb
        }
    }
}

type NList<T> = NodeList<T, TopologyType>;
type EList<T> = EdgeList<T, TopologyType>;
type Failures = SmallVec<[(RouterId, RouterId); 4]>;

#[allow(missing_docs)]
#[derive(Debug)]
pub struct WorstCaseState<P> {
    pub routing_inputs: Option<HashMap<P, WorstCasePrefixState>>,
    pub failures: Failures,
}

#[derive(Debug, NetworkFormatter)]
/// The worst-case state for a speficic prefix.
pub enum WorstCasePrefixState {
    /// The worst case set of egress routers. This variant is only returned if the prefix's egress
    /// points can change.
    Egresses(Vec<RouterId>),
    /// The worst case ingress point, where traffic entering causes the highest link load. This
    /// variant is only returned if the prefix's ingress point can change.
    Ingress(RouterId),
    /// The variant where both the ingress and egress points are changed. This variant is only
    /// returned for prefixes where both the ingress and egress can change.
    Both {
        /// Worst-case ingress
        ingress: RouterId,
        /// Worst-case egress
        egress: RouterId,
    },
}

/// The result obtained from Velo
#[derive(Debug)]
pub struct PerformanceReport<P> {
    /// Positive error bounds. The actual maximum-load may be higher by `pos_bound`.
    pub pos_bounds: f64,
    /// Negative error bounds. The actual maximum-load may be lower by `neg_bound`.
    pub neg_bounds: f64,
    /// Actual loads
    pub loads: HashMap<(RouterId, RouterId), f64>,
    /// The worst-case state for each link
    pub states: HashMap<(RouterId, RouterId), WorstCaseState<P>>,
}

/// A velo analysis structure with prepared data
#[derive(Debug)]
pub struct VeloAnalysis<'a, P> {
    config: &'a Velo<P>,
    /// The data to analyze
    pub(crate) data: Vec<PrefixData<'a, P>>,
    /// Error (underapproximation)
    error_pos: f64,
    /// Error (overapproximation)
    error_neg: f64,
}

impl<'a, P: Prefix + Sync + Send + 'static> VeloAnalysis<'a, P> {
    fn new(
        config: &'a Velo<P>,
        current_state: &'a HashMap<P, Vec<RouterId>>,
        traffic_matrix: &'a TrafficMatrix<P>,
    ) -> Self {
        let pecs = pecs::find(config, current_state, traffic_matrix);
        let (mut data, error) = cluster(config, pecs);

        // sort the data by total demand
        data.sort_by(|a, b| b.total_demand.partial_cmp(&a.total_demand).unwrap());
        let mut error_pos = error / 2.0;
        let error_neg = error / 2.0;

        // only take the first few
        if let Some(k) = config.heaviest_pecs {
            if k < data.len() {
                error_pos += data[k..].iter().map(|x| x.total_demand).sum::<f64>();
                data.truncate(k);
            }
        }

        Self {
            config,
            data,
            error_pos,
            error_neg,
        }
    }

    /// Return the total demand present in the analysis
    pub fn total_demand(&self) -> f64 {
        self.data.iter().map(|d| d.total_demand).sum()
    }

    /// Get the number of clusters (prefixes that are part of the computation)
    pub fn num_clusters(&self) -> usize {
        self.data.len()
    }

    /// Compute the current link load.
    pub fn current_link_load(&self) -> HashMap<(RouterId, RouterId), f64> {
        let mut dags = self.compute_dags(&[]);
        let mut loads: EList<f64> = EList::new(&self.config.topo);

        // go through each prefix
        for egress_data in &self.data {
            // get the current demand
            let current = egress_data.get_current_state(&self.config.topo, &mut dags, false);
            loads.zip_mut(&current, |cur, new| *cur += *new);
        }

        loads
            .into_idx_val()
            .map(|(e, load)| (self.config.topo.net_link(e), load))
            .collect()
    }

    /// Get the k prefixes that carry the most traffic.
    pub fn top_k_prefixes(&self, k: usize) -> Vec<P> {
        if k == 0 {
            return Vec::new();
        }

        // a heap to store the ne
        let mut heap = std::collections::BinaryHeap::<(NotNan<f64>, P)>::new();
        let mut min = 0.0;
        for &(p, x) in self.data.iter().flat_map(|d| &d.prefixes) {
            if min < x {
                heap.push((NotNan::new(-x).unwrap(), p));
                if heap.len() > k {
                    let (new_min, _) = heap.pop().unwrap();
                    min = -new_min.into_inner();
                }
            }
        }

        let mut prefixes = heap.into_iter().take(k).map(|(_, p)| p).collect::<Vec<_>>();
        prefixes.reverse();
        prefixes
    }

    /// install the given TE paths for all prerfixes. This function must only be used for evaluation!
    pub fn install_te_paths(
        &mut self,
        te_paths: &'a BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<EdgeId>>>>,
    ) {
        for x in self.data.iter_mut() {
            x.te_paths = Some(te_paths);
        }
    }

    /// Install the given TE paths for the i'th prefix equivalence class that is of kind
    /// `Uncertainty::Egress`. This function must only be used for evaluation!
    pub fn install_te_paths_at(
        &mut self,
        i: usize,
        te_paths: &'a BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<EdgeId>>>>,
    ) {
        if let Some(x) = self.data.iter_mut().filter(|x| x.uncertainty.only_egress()).nth(i) {
            x.te_paths = Some(te_paths);
        }
    }
}

#[derive(Debug)]
/// Data for a set of prefixes that all have the same configuration, and that have a similar traffic
/// distribution.
pub(crate) struct PrefixData<'a, P> {
    /// The (currnet, normalized) demand, i.e., traffic entering the network. In case this element
    /// describes multiple prefixes, then this current_demand is the sum of all demands for all
    /// prefixes, normalized to sum up to 1.
    current_norm_demand: NList<f64>,
    /// The (current) set of egress routers used for routing.
    current_state: &'a BTreeSet<NodeId>,
    /// The set of prefixes in that class, including the total traffic towards that prefix.
    pub(crate) prefixes: Vec<(P, f64)>,
    /// The total traffic, i.e., the sum of traffic towards all prefixes.
    total_demand: f64,
    /// All traffic engineering paths configured for that set of prefixes.
    te_paths: Option<&'a BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<EdgeId>>>>>,
    /// The uncertainty mode.
    uncertainty: &'a TUncertainty,
}

const EMPTY_MAP: &BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<EdgeId>>>> = &BTreeMap::new();

impl<'a, P> PrefixData<'a, P> {
    fn get_te_paths(&self) -> &BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<EdgeId>>>> {
        self.te_paths.unwrap_or(EMPTY_MAP)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// The uncertainty to handle the prefix. This is the same as `Uncertainty`, but all router IDs
/// translated into the topology view.
pub(crate) enum TUncertainty {
    /// We can only vary the set of egress points.
    Egress {
        /// A set of possible egress points, together with a set of possible local-pref values,
        /// with which a route at that egress may be advertised.
        egresses: BTreeMap<NodeId, BTreeSet<u32>>,
    },
    /// We can only vary the set of ingress points
    Ingress {
        /// A set of possible ingress points.
        ingresses: BTreeSet<NodeId>,
    },
    /// We can vary both the ingress and the egress point.
    Both {
        /// A set of possible egress points, together with a set of possible local-pref values,
        /// with which a route at that egress may be advertised.
        egresses: BTreeMap<NodeId, BTreeSet<u32>>,
        /// A set of possible ingress points.
        ingresses: BTreeSet<NodeId>,
    },
}

impl TUncertainty {
    /// Return `true` only if `self` is of type `TUncertainty::Egress`.
    pub(self) fn only_egress(&self) -> bool {
        matches!(self, TUncertainty::Egress { .. })
    }
}

/// Settings for a specific prefix, used as an intermediate representation before clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PrefixSettings<'a> {
    /// The set of traffic-engineering paths that are configured.
    te_paths: Option<&'a BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<EdgeId>>>>>,
    /// The current routing state used for routing
    current_state: &'a BTreeSet<NodeId>,
    /// The uncertainty mode for all those prefixes
    uncertainty: &'a TUncertainty,
}
