//! Module to combine equivalence classes of destination prefixes.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Display;

use crate::MyProgressIterator;

use super::clustering::*;
use super::TrafficMatrix;

use bgpsim::types::{Prefix, RouterId, SimplePrefix};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use strum::{EnumIter, IntoEnumIterator};

/// An approximated traffic matrix with less number of prefixes
#[derive(Debug)]
pub struct ApproxTM<'a, P> {
    /// The original traffic matrix
    pub original: &'a TrafficMatrix<P>,
    /// The approximated traffic matrix
    pub approx: TrafficMatrix<SimplePrefix>,
    /// Total amount of demand in the original traffic matrix.
    pub total_demand: f64,
    /// Total amount of demand in the approximated traffic matrix.
    pub approx_demand: f64,
    /// A lookup to associate each original prefix (`P`) with its equivalence class
    /// (`SimplePrefix`).
    pub lookup: HashMap<P, SimplePrefix>,
    /// A mapping of each equivalence class (`SimplePrefix`) to a list of original prefixes (`P`)
    /// that are summarized by that equivalence class.
    pub reverse: HashMap<SimplePrefix, Vec<P>>,
    /// The positive error, that is, traffic that might not be considered during the analysis.
    pub error_pos: f64,
    /// The negative error, that is, traffic that might be unnecessarily considered during the
    /// analysis.
    pub error_neg: f64,
}

impl<'a, P: Prefix> ApproxTM<'a, P> {
    /// Generate the approximated TrafficMatrix from a TrafficMatrix data.
    pub fn from_data<C: Clustering>(
        data: &TrafficMatrixData<P>,
        original: &'a TrafficMatrix<P>,
        algo: C,
        progress: bool,
    ) -> Self {
        let mut clustering = algo.cluster(data, progress);
        let mode = algo.mode();

        let total_demand = data.total_demand;
        let mut approx_demand = 0.0;

        let error_pos = clustering.error_pos(mode) * data.total_demand;
        let error_neg = clustering.error_neg(mode) * data.total_demand;

        // construct the data
        let mut lookup: HashMap<P, SimplePrefix> = HashMap::new();
        let mut reverse: HashMap<SimplePrefix, Vec<P>> = HashMap::new();
        let mut approx = TrafficMatrix::new();
        for (i, mut cluster) in clustering.clusters.into_iter().enumerate() {
            // generate a prefix equivalence class
            let pec = SimplePrefix::from(i as u32);
            // construct the resulting traffic and add it to the matrix
            let demand = cluster.effective_demand(mode);
            for (&src, &i_src) in data.source_lut.iter() {
                if let Some(&d) = demand.get(i_src) {
                    approx.insert(crate::performance::TrafficClass { src, dst: pec }, d);
                    approx_demand += d;
                }
            }
            // construct the lookup table
            for member in &cluster.members {
                lookup.insert(member.prefix, pec);
                reverse.entry(pec).or_default().push(member.prefix);
            }
        }

        Self {
            original,
            approx,
            lookup,
            reverse,
            error_pos,
            error_neg,
            total_demand,
            approx_demand,
        }
    }

    /// Create a new approximated traffic matrix using the clustering algorithm `algo`.
    pub fn new<C: Clustering>(original: &'a TrafficMatrix<P>, algo: C, progress: bool) -> Self {
        let data = TrafficMatrixData::new(original, progress);
        Self::from_data(&data, original, algo, progress)
    }
}

impl<'a> ApproxTM<'a, SimplePrefix> {
    /// Create an empty ApproxTM
    pub fn empty(original: &'a TrafficMatrix<SimplePrefix>) -> Self {
        let lookup = original.keys().map(|x| (x.dst, x.dst)).collect::<HashMap<_, _>>();
        let reverse = lookup.values().map(|p| (*p, vec![*p])).collect();
        let total_demand = original.values().sum::<f64>();
        Self {
            original,
            approx: original.clone(),
            total_demand,
            approx_demand: total_demand,
            lookup,
            reverse,
            error_pos: 0.0,
            error_neg: 0.0,
        }
    }
}

/// Score the clustering approaches on the given traffic matrix
pub fn score_clustering<P: Prefix + Sync>(tm: &TrafficMatrix<P>) {
    let data = TrafficMatrixData::new(tm, false);
    data.print_statistics();

    for mode in ClusterMode::iter() {
        SingleCluster { mode }
            .cluster(&data, false)
            .print_score(mode, format!("single({mode:?})"));
    }

    for k in [10, 50, 100, 500, 1000] {
        for mode in ClusterMode::iter() {
            HeaviestK { k, mode }
                .cluster(&data, false)
                .print_score(mode, format!("heaviest(k={k}, {mode:?})"));
        }
    }

    for k in [10, 50, 100] {
        for mode in ClusterMode::iter() {
            let kmeans = KMeans {
                k,
                mode,
                ..Default::default()
            };
            kmeans
                .cluster(&data, false)
                .print_score(mode, format!("KMeans(k={k}, {mode:?})"));
        }
    }
}

/// Dataset that stores the traffic matrix in a more useful format.
#[derive(Debug)]
#[allow(dead_code)]
pub struct TrafficMatrixData<P: Prefix, ID = RouterId> {
    /// Number of prefixes (datapoints)
    pub num_prefixes: usize,
    /// Number of sources (dimensions)
    pub num_sources: usize,
    /// Lookup table to assign each source router to a given dimension
    pub source_lut: BTreeMap<ID, usize>,
    /// Lookup to assign each prefix to a single datapoint
    pub prefix_lut: BTreeMap<P, usize>,
    /// Total amount of traffic
    pub total_demand: f64,
    /// The actual data.
    pub data: Vec<PerPrefixDemand<P>>,
}

impl<P: Prefix> TrafficMatrixData<P, RouterId> {
    /// Create a new `TrafficMatrixData` from a `TrafficMatrix`.
    pub fn new(tm: &TrafficMatrix<P>, progress: bool) -> Self {
        let mut prefixes = BTreeSet::new();
        let mut sources = BTreeSet::new();
        for tc in tm.keys().my_progress("Preparing traffic matrix [1]", false, progress) {
            sources.insert(tc.src);
            prefixes.insert(tc.dst);
        }
        let mut total_demand = 0.0;

        // create the lookup for routers
        let source_lut: BTreeMap<RouterId, usize> =
            sources.iter().enumerate().map(|(i, r)| (*r, i)).collect();
        let prefix_lut: BTreeMap<P, usize> =
            prefixes.iter().enumerate().map(|(i, p)| (*p, i)).collect();

        // initialize the data
        let mut data: Vec<PerPrefixDemand<P>> = prefixes
            .iter()
            .map(|&prefix| PerPrefixDemand {
                prefix,
                demand: vec![0.0; sources.len()],
                norm_demand: vec![0.0; sources.len()],
                scale: 1.0,
            })
            .collect();

        // assign the values
        for (tc, demand) in tm.iter().my_progress("Preparing traffic matrix [2]", false, progress) {
            let p = prefix_lut[&tc.dst];
            let i = source_lut[&tc.src];
            data[p].demand[i] = *demand;
            total_demand += demand;
        }

        // normalize the data
        data.iter_mut()
            .my_progress("Preparing traffic matrix [3]", false, progress)
            .for_each(|d| d.normalize());

        Self {
            num_prefixes: prefixes.len(),
            num_sources: sources.len(),
            total_demand,
            source_lut,
            prefix_lut,
            data,
        }
    }
}

impl<P: Prefix, ID> TrafficMatrixData<P, ID> {
    /// Print the statistics of the traffic matrix
    pub fn print_statistics(&self) {
        println!("total demand: {}", self.total_demand);
        let sorted_scale = self
            .data
            .iter()
            .map(|x| ordered_float::NotNan::new(x.scale / self.total_demand).unwrap())
            .sorted()
            .map(|x| x.into_inner() * 100.0)
            .collect_vec();

        println!(
            "       top 1: {: >5.2}%",
            sorted_scale.last().copied().unwrap_or_default()
        );
        println!(
            "      top 10: {: >5.2}%",
            sorted_scale.iter().rev().take(10).sum::<f64>()
        );
        println!(
            "     top 100: {: >5.2}%",
            sorted_scale.iter().rev().take(100).sum::<f64>()
        );
        println!(
            "      top 1%: {: >5.2}%",
            sorted_scale.iter().rev().take(1000).sum::<f64>()
        );
        println!(
            "     top 10%: {: >5.2}%",
            sorted_scale.iter().rev().take(10000).sum::<f64>()
        );
        println!(
            "     top 15%: {: >5.2}%",
            sorted_scale.iter().rev().take(15000).sum::<f64>()
        );
    }
}

/// A single datapoint, i.e., the demand for a single prefix.
#[derive(Debug, Clone)]
pub struct PerPrefixDemand<P> {
    /// The associated prefix
    pub prefix: P,
    /// The vectorized demand
    pub demand: Vec<f64>,
    /// The normalized demand, such that all demands sum up to exactly 1.
    pub norm_demand: Vec<f64>,
    /// the scaling factor of that point
    pub scale: f64,
}

impl<P> PerPrefixDemand<P> {
    /// Initialize `self.norm_demand` and `self.scale`.
    pub fn normalize(&mut self) {
        (self.norm_demand, self.scale) = normalize(&self.demand);
    }

    /// Compare `self.demand` (or `self.norm_demand` if `normalized` is `true`) with `other`,
    /// applying `f` to each pair, and summing up the result. The function `op` is called with the
    /// first argument being the respective demand, while other is the value of `other`.
    fn compare<F: FnMut(f64, f64) -> f64>(
        &self,
        normalized: bool,
        other: &[f64],
        mut op: F,
    ) -> f64 {
        let x = if normalized {
            &self.norm_demand
        } else {
            &self.demand
        };
        x.iter().zip(other.iter()).map(|(a, b)| op(*a, *b)).sum()
    }

    /// Compute the positive error, that is, traffic that might not be considered during the
    /// analysis.
    pub fn error_pos(&self, center: &[f64], mode: ClusterMode) -> f64 {
        let cost = self.compare(mode.normalized(), center, |x, r| (x - r).max(0.0));
        let scale = if mode.normalized() { self.scale } else { 1.0 };
        cost * scale
    }

    /// Compute the negative error, that is, traffic that might be unnecessarily considered during
    /// the analysis.
    pub fn error_neg(&self, center: &[f64], mode: ClusterMode) -> f64 {
        let cost = self.compare(mode.normalized(), center, |x, r| (r - x).max(0.0));
        let scale = if mode.normalized() { self.scale } else { 1.0 };
        cost * scale
    }
}

pub(super) fn normalize(demand: &[f64]) -> (Vec<f64>, f64) {
    let scale: f64 = demand.iter().sum();
    if scale == 0.0 {
        (demand.to_vec(), 1.0)
    } else {
        let inv_scale = 1.0 / scale;
        let normalized = demand.iter().map(|&x| x * inv_scale).collect();
        (normalized, scale)
    }
}

/// Enumeration of different kinds of clustering approaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumIter, Deserialize, Serialize)]
pub enum ClusterMode {
    /// Minimize the (non-normalized) distance.
    Direct,
    /// Minimize the normalized distance (without scaling).
    Normalized,
    /// Minimize the normalized distance with scaling.
    NormalizedScaled,
}

impl ClusterMode {
    /// Whether the data is normalized for clustering.
    pub fn normalized(&self) -> bool {
        matches!(self, Self::Normalized | Self::NormalizedScaled)
    }
}

/// Representation of a single cluster
#[derive(Debug, Clone)]
pub struct Cluster<'a, P> {
    point: Option<(ClusterMode, Vec<f64>)>,
    members: Vec<&'a PerPrefixDemand<P>>,
}

impl<'a, P> Cluster<'a, P> {
    fn from_members<I: IntoIterator<Item = &'a PerPrefixDemand<P>>>(members: I) -> Self {
        Self {
            point: None,
            members: members.into_iter().collect(),
        }
    }

    /// Get a reference to all members
    pub fn members(&self) -> &[&'a PerPrefixDemand<P>] {
        &self.members
    }

    /// Compute the midpoint of a clustering if necessary and return a reference to it.
    pub fn compute_point(&mut self, mode: ClusterMode) -> &[f64] {
        // do nothing if there are no members.
        if self.members.is_empty() {
            return &[];
        }
        // Don't recompute the point if is is already computed
        if self.point.as_ref().map(|(m, _)| m) == Some(&mode) {
            return &self.point.as_ref().unwrap().1;
        }

        let num_members = self.members.len();
        let dim = self.members.first().unwrap().demand.len();
        let point: Vec<f64> = match mode {
            ClusterMode::Direct => (0..dim)
                .map(|i| {
                    self.members.iter().map(|m| m.demand[i]).sum::<f64>() / (num_members as f64)
                })
                .collect(),
            ClusterMode::Normalized => (0..dim)
                .map(|i| {
                    self.members.iter().map(|m| m.norm_demand[i]).sum::<f64>()
                        / (num_members as f64)
                })
                .collect(),
            ClusterMode::NormalizedScaled => {
                let total_scale = self.members.iter().map(|x| x.scale).sum::<f64>();
                (0..dim)
                    .map(|i| self.members.iter().map(|m| m.demand[i]).sum::<f64>() / total_scale)
                    .collect()
            }
        };
        self.point = Some((mode, point));
        &self.point.as_ref().unwrap().1
    }

    /// Compute the positive error, that is, traffic that might not be considered during the
    /// analysis.
    pub fn error_pos(&mut self, mode: ClusterMode) -> f64 {
        if self.members.is_empty() {
            return 0.0;
        }
        // recompute the point if necessary
        self.compute_point(mode);
        let point = &self.point.as_ref().unwrap().1;

        self.members.iter().map(|m| m.error_pos(point, mode)).sum()
    }

    /// Compute the negative error, that is, traffic that might be unnecessarily considered during
    /// the analysis.
    pub fn error_neg(&mut self, mode: ClusterMode) -> f64 {
        if self.members.is_empty() {
            return 0.0;
        }
        // recompute the point if necessary
        self.compute_point(mode);
        let point = &self.point.as_ref().unwrap().1;

        self.members.iter().map(|m| m.error_neg(point, mode)).sum()
    }

    /// Compute the effective demand of that cluster.
    pub fn effective_demand(&mut self, mode: ClusterMode) -> Vec<f64> {
        if self.members.is_empty() {
            return Vec::new();
        }
        // recompute the point if necessary
        self.compute_point(mode);
        let point = &self.point.as_ref().unwrap().1;

        let factor = match mode {
            ClusterMode::Direct => self.members.len() as f64,
            ClusterMode::Normalized | ClusterMode::NormalizedScaled => {
                self.members.iter().map(|m| m.scale).sum::<f64>()
            }
        };
        point.iter().map(|x| *x * factor).collect()
    }
}

/// A clustering for the given dataset.
#[derive(Debug)]
pub struct Clusters<'a, P> {
    /// Vector oif all clusters
    pub clusters: Vec<Cluster<'a, P>>,
    /// Total demand of all prefixes.
    pub total_demand: f64,
}

impl<'a, P: Prefix> Clusters<'a, P> {
    /// Create the clustering from a prediction.
    pub fn from_prediction<ID>(data: &'a TrafficMatrixData<P, ID>, result: &[usize]) -> Self {
        assert_eq!(result.len(), data.num_prefixes);
        let num_clusters = result.iter().copied().max().unwrap_or(0) + 1;
        let clusters = (0..num_clusters)
            .map(|c| {
                result
                    .iter()
                    .enumerate()
                    .filter(move |(_, x)| **x == c)
                    .map(|(i, _)| &data.data[i])
            })
            .map(Cluster::from_members)
            .collect();
        Self {
            clusters,
            total_demand: data.total_demand,
        }
    }

    /// Compute the positive error, that is, traffic that might not be considered during the
    /// analysis.
    pub fn error_pos(&mut self, mode: ClusterMode) -> f64 {
        self.clusters.iter_mut().map(|c| c.error_pos(mode)).sum::<f64>() / self.total_demand
    }

    /// Compute the negative error, that is, traffic that might be unnecessarily considered during
    /// the analysis.
    pub fn error_neg(&mut self, mode: ClusterMode) -> f64 {
        self.clusters.iter_mut().map(|c| c.error_neg(mode)).sum::<f64>() / self.total_demand
    }

    /// Print the score of the clustering
    pub fn print_score(&mut self, mode: ClusterMode, name: impl Display) {
        let error_pos = self.error_pos(mode);
        let error_neg = self.error_neg(mode);
        approx::assert_relative_eq!(error_pos, error_neg, epsilon = 1e-8);
        println!(
            "{name: <35}: ±{:.5} (n={}, mode={mode:?})",
            error_pos,
            self.clusters.len()
        );
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use bgpsim::types::SimplePrefix;
    use maplit::hashmap;

    use crate::performance::TrafficClass;

    use super::*;

    type P = SimplePrefix;

    macro_rules! assert_vec_sim {
        ($left:expr, $right:expr) => {
            let left = $left;
            let right = $right;
            for (i, (l, r)) in left.into_iter().zip(right).enumerate() {
                if !approx::relative_eq!(*l, *r) {
                    panic!("Vector different at {i}:\n\n    left  = {l}\n    right = {r}\n\n    left vector  = {left:?}\n    right vector = {right:?}\n")
                }
            }
        };
    }

    #[test]
    fn normalize() {
        let tm: TrafficMatrix<P> = hashmap! {
            TrafficClass { src: 0.into(), dst: 0.into() } => 1.0,
            TrafficClass { src: 1.into(), dst: 0.into() } => 10.0,
            TrafficClass { src: 0.into(), dst: 1.into() } => 100.0,
            TrafficClass { src: 1.into(), dst: 1.into() } => 10.0,
        };

        let data = TrafficMatrixData::new(&tm, false);

        for d in data.data {
            d.demand
                .iter()
                .copied()
                .zip(d.norm_demand.iter().map(|x| *x * d.scale))
                .all(|(a, b)| a == b);
            assert_eq!(d.norm_demand.iter().copied().sum::<f64>(), 1.0);
        }
    }

    #[test]
    fn clustering_direct() {
        let tm: TrafficMatrix<P> = hashmap! {
            TrafficClass { src: 0.into(), dst: 0.into() } => 1.0,
            TrafficClass { src: 1.into(), dst: 0.into() } => 10.0,
            TrafficClass { src: 0.into(), dst: 1.into() } => 100.0,
            TrafficClass { src: 1.into(), dst: 1.into() } => 10.0,
        };

        let data = TrafficMatrixData::new(&tm, false);

        let mut clustering = Clusters::from_prediction(&data, &[0, 0]);
        let c = clustering.clusters.get_mut(0).unwrap();
        let m = ClusterMode::Direct;
        assert_vec_sim!(c.compute_point(m), &[50.5, 10.0]);
        assert_relative_eq!(c.error_pos(m), c.error_neg(m));
    }

    #[test]
    fn clustering_norm_perfect() {
        let tm: TrafficMatrix<P> = hashmap! {
            TrafficClass { src: 0.into(), dst: 0.into() } => 9.0,
            TrafficClass { src: 1.into(), dst: 0.into() } => 1.0,
            TrafficClass { src: 0.into(), dst: 1.into() } => 90.0,
            TrafficClass { src: 1.into(), dst: 1.into() } => 10.0,
        };

        let data = TrafficMatrixData::new(&tm, false);

        let mut clustering = Clusters::from_prediction(&data, &[0, 0]);
        let c = clustering.clusters.get_mut(0).unwrap();
        assert_vec_sim!(c.compute_point(ClusterMode::Normalized), &[0.9, 0.1]);
        assert_vec_sim!(c.compute_point(ClusterMode::NormalizedScaled), &[0.9, 0.1]);
    }

    #[test]
    fn clustering_norm_almost_perfect() {
        let tm: TrafficMatrix<P> = hashmap! {
            TrafficClass { src: 0.into(), dst: 0.into() } => 9.1,
            TrafficClass { src: 1.into(), dst: 0.into() } => 0.9,
            TrafficClass { src: 0.into(), dst: 1.into() } => 89.0,
            TrafficClass { src: 1.into(), dst: 1.into() } => 11.0,
        };

        let data = TrafficMatrixData::new(&tm, false);

        let mut clustering = Clusters::from_prediction(&data, &[0, 0]);
        let c = clustering.clusters.get_mut(0).unwrap();
        assert_relative_eq!(
            c.error_pos(ClusterMode::Direct),
            c.error_neg(ClusterMode::Direct)
        );
        let p_norm = c.compute_point(ClusterMode::Normalized).to_vec();
        let p_scaled = c.compute_point(ClusterMode::NormalizedScaled).to_vec();
        assert_relative_eq!(p_norm.iter().copied().sum::<f64>(), 1.0);
        assert_relative_eq!(p_scaled.iter().copied().sum::<f64>(), 1.0);
        println!("{p_norm:?}");
        println!("{p_scaled:?}");
        assert_vec_sim!(&p_norm, &[0.9, 0.1]);
        assert!(data.data[1].norm_demand[0] < p_scaled[0] && p_scaled[0] < p_norm[0]);
        assert!(data.data[1].norm_demand[1] > p_scaled[1] && p_scaled[1] > p_norm[1]);
    }

    #[test]
    fn clustering_norm_almost_perfect_extreme() {
        let tm: TrafficMatrix<P> = hashmap! {
            TrafficClass { src: 0.into(), dst: 0.into() } => 9.0001,
            TrafficClass { src: 1.into(), dst: 0.into() } => 0.9999,
            TrafficClass { src: 0.into(), dst: 1.into() } => 89999.0,
            TrafficClass { src: 1.into(), dst: 1.into() } => 10001.0,
        };

        let data = TrafficMatrixData::new(&tm, false);

        let mut clustering = Clusters::from_prediction(&data, &[0, 0]);
        let c = clustering.clusters.get_mut(0).unwrap();
        assert_relative_eq!(
            c.error_pos(ClusterMode::Direct),
            c.error_neg(ClusterMode::Direct)
        );
        let p_norm = c.compute_point(ClusterMode::Normalized).to_vec();
        let p_scaled = c.compute_point(ClusterMode::NormalizedScaled).to_vec();
        assert_relative_eq!(p_norm.iter().copied().sum::<f64>(), 1.0);
        assert_relative_eq!(p_scaled.iter().copied().sum::<f64>(), 1.0);
        println!("{p_norm:?}");
        println!("{p_scaled:?}");
        assert_vec_sim!(&p_norm, &[0.9, 0.1]);
        assert!(data.data[1].norm_demand[0] < p_scaled[0] && p_scaled[0] < p_norm[0]);
        assert!(data.data[1].norm_demand[1] > p_scaled[1] && p_scaled[1] > p_norm[1]);
    }

    #[test]
    fn clustering_norm_almost_perfect_3x3() {
        let tm: TrafficMatrix<P> = hashmap! {
            TrafficClass { src: 0.into(), dst: 0.into() } => 8.1,
            TrafficClass { src: 1.into(), dst: 0.into() } => 0.9,
            TrafficClass { src: 2.into(), dst: 0.into() } => 1.0,
            TrafficClass { src: 0.into(), dst: 1.into() } => 79.5,
            TrafficClass { src: 1.into(), dst: 1.into() } => 10.5,
            TrafficClass { src: 2.into(), dst: 1.into() } => 10.0,
            TrafficClass { src: 0.into(), dst: 2.into() } => 795.0,
            TrafficClass { src: 1.into(), dst: 2.into() } => 105.0,
            TrafficClass { src: 2.into(), dst: 2.into() } => 100.0,
        };

        let data = TrafficMatrixData::new(&tm, false);

        let mut clustering = Clusters::from_prediction(&data, &[0, 0, 0]);
        let c = clustering.clusters.get_mut(0).unwrap();
        assert_relative_eq!(
            c.error_pos(ClusterMode::Direct),
            c.error_neg(ClusterMode::Direct)
        );
        let p_norm = c.compute_point(ClusterMode::Normalized).to_vec();
        let p_scaled = c.compute_point(ClusterMode::NormalizedScaled).to_vec();
        assert_relative_eq!(p_norm.iter().copied().sum::<f64>(), 1.0);
        assert_relative_eq!(p_scaled.iter().copied().sum::<f64>(), 1.0);
        println!("{p_norm:?}");
        println!("{p_scaled:?}");
        assert_vec_sim!(&p_norm, &[0.8, 0.1, 0.1]);
        assert!(data.data[2].norm_demand[0] < p_scaled[0] && p_scaled[0] < p_norm[0]);
        assert!(data.data[2].norm_demand[1] > p_scaled[1] && p_scaled[1] > p_norm[1]);
        assert_relative_eq!(data.data[2].norm_demand[2], p_scaled[2]);
        assert_relative_eq!(p_scaled[2], p_norm[2]);
    }
}
