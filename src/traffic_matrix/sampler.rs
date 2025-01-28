//! Methods for sampling traffic matrices

use std::collections::{BTreeMap, HashMap};

use bgpsim::{
    network::Network,
    types::{Prefix, RouterId},
};
use itertools::Itertools;
use rand::prelude::*;
use rand_distr::{Beta, Exp, LogNormal, Normal, Pareto, Poisson, Uniform};

use crate::{algorithms::Topology, performance::TrafficClass, MyProgressIterator};

use super::TrafficMatrix;

/// A generalized traffic matrix sampler interface.
pub trait TMSampler<P: Prefix> {
    /// Create a new TMSampler
    fn new<Q>(net: &Network<P, Q>) -> Self;

    /// Set the number of prefixes
    fn prefixes(&mut self, num: u32) -> &mut Self;

    /// Set the seed to get a predictable outcome.
    fn seed(&mut self, seed: u64) -> &mut Self;

    /// Sample one traffic matrix.
    fn sample(&mut self, progress: bool) -> TrafficMatrix<P>;

    /// Sample a random current state
    fn current_state(
        &mut self,
        num_states: Option<usize>,
        progress: bool,
    ) -> HashMap<P, Vec<RouterId>>;

    /// Sample the Traffic Engineering pahts
    fn te_paths(
        &mut self,
        num_paths_per_prefix: usize,
    ) -> Vec<(RouterId, RouterId, Vec<Vec<RouterId>>)>;
}

/// Create a gravity model for sampling traffic matrices. The traffic matrix will have size `p^2`,
/// where `p` is the number of prefixes. In the end, the traffic for each source prefix will be
/// assigned randomly to one of the source routers.
#[derive(Debug)]
pub struct GravityTMSampler<P, Rep, Att, Fri> {
    /// The stored topology for sampling TE paths
    topo: Topology,
    /// Number of prefixes to include in the matrix. (10 by default)
    num_prefixes: u32,
    /// Routers in the network
    sources: Vec<RouterId>,
    /// Border routers in the network
    egresses: Vec<RouterId>,
    /// External routers in the network
    externals: Vec<RouterId>,
    /// Seed to use
    seed: Option<u64>,
    /// Distribution for the repulsive factor
    repulsion: Option<Rep>,
    /// Distribution for the attractive factor
    attraction: Option<Att>,
    /// Distribution for the friction factor
    friction: Option<Fri>,
    /// Data for the sampled repulsion
    sampled_repulsion: BTreeMap<RouterId, f64>,
    /// Data for the sampled attraction
    sampled_attraction: BTreeMap<P, f64>,
    /// If set to true, the model will multiply by the friction, instead of dividing.
    invert_friction: bool,
    /// If set to true, the model will re-normalize by attraction, to get the same numbers as
    /// expected.
    renormalize_attraction: bool,
}

impl<P: Prefix, Rep, Att, Fri> TMSampler<P> for GravityTMSampler<P, Rep, Att, Fri>
where
    Rep: Distribution<f64>,
    Att: Distribution<f64>,
    Fri: Distribution<f64>,
{
    fn new<Q>(net: &Network<P, Q>) -> Self {
        Self {
            topo: Topology::new(net),
            num_prefixes: 10,
            sources: net.internal_indices().sorted().collect(),
            egresses: net
                .ospf_network()
                .external_edges()
                .map(|e| e.int)
                .unique()
                .sorted()
                .collect(),
            externals: net.external_indices().sorted().collect(),
            seed: None,
            repulsion: None,
            attraction: None,
            friction: None,
            sampled_repulsion: Default::default(),
            sampled_attraction: Default::default(),
            invert_friction: false,
            renormalize_attraction: false,
        }
    }

    fn prefixes(&mut self, num: u32) -> &mut Self {
        self.num_prefixes = num;
        self
    }

    fn seed(&mut self, seed: u64) -> &mut Self {
        self.seed = Some(seed);
        self
    }

    fn sample(&mut self, progress: bool) -> TrafficMatrix<P> {
        let mut rng = if let Some(seed) = self.seed {
            SmallRng::seed_from_u64(seed)
        } else {
            SmallRng::from_entropy()
        };

        let prefixes = (0..self.num_prefixes).map(P::from).collect::<Vec<_>>();
        let norm = self.sources.len() as f64 / prefixes.len() as f64;

        let d_rep = self.repulsion.as_ref().expect("Repulsion dist is not set!");
        let d_att = self.attraction.as_ref().expect("Attraction dist is not set!");
        let d_fri = self.friction.as_ref().expect("Friction dist is not set!");

        self.sampled_repulsion = self
            .sources
            .iter()
            .my_progress("Sampling the traffic matrix (repulsion)", false, progress)
            .map(|r| (*r, d_rep.sample(&mut rng)))
            .collect();
        self.sampled_attraction = prefixes
            .iter()
            .my_progress("Sampling the traffic matrix (attraction)", false, progress)
            .map(|p| (*p, d_att.sample(&mut rng)))
            .collect();

        // statistics needed to renormalize attraction
        let mut total_traffic: f64 = 0.0;
        let mut traffic_per_dst: HashMap<P, f64> = HashMap::new();

        let mut result: TrafficMatrix<P> = HashMap::new();
        for (src, rep) in self.sampled_repulsion.iter().my_progress(
            "Sampling the traffic matrix (friction)",
            false,
            progress,
        ) {
            let (src, rep) = (*src, *rep);
            for (dst, att) in self.sampled_attraction.iter() {
                let dst = *dst;
                let tc = TrafficClass { src, dst };
                let fri = d_fri.sample(&mut rng);
                let demand = if self.invert_friction {
                    rep * att * fri
                } else {
                    rep * att / fri
                };
                let demand = demand * norm;
                total_traffic += demand;
                *traffic_per_dst.entry(dst).or_default() += demand;
                result.insert(tc, demand);
            }
        }

        // renormalize if requested
        if self.renormalize_attraction {
            let total_attraction: f64 = self.sampled_attraction.values().copied().sum::<f64>();
            for (tc, demand) in result.iter_mut() {
                let want_prop = self.sampled_attraction[&tc.dst] / total_attraction;
                let got_prop = traffic_per_dst[&tc.dst] / total_traffic;
                let scale = want_prop / got_prop;
                *demand *= scale;
            }
        }

        result
    }

    fn current_state(
        &mut self,
        num_states: Option<usize>,
        progress: bool,
    ) -> HashMap<P, Vec<RouterId>> {
        if let Some(num_states) = num_states {
            random_distinct_current_state(
                &self.externals,
                (0..self.num_prefixes).map(P::from),
                num_states,
                self.seed,
                progress,
            )
        } else {
            random_current_state(
                &self.externals,
                (0..self.num_prefixes).map(P::from),
                self.seed,
                progress,
            )
        }
    }

    fn te_paths(&mut self, num_paths: usize) -> Vec<(RouterId, RouterId, Vec<Vec<RouterId>>)> {
        let mut rng = if let Some(seed) = self.seed {
            SmallRng::seed_from_u64(seed)
        } else {
            SmallRng::from_entropy()
        };
        let sources = self.sampled_repulsion.iter().map(|(s, x)| (*s, *x)).collect_vec();

        let mut te_paths: Vec<(RouterId, RouterId, Vec<Vec<RouterId>>)> = Vec::new();
        for _ in 0..num_paths {
            loop {
                let (ingress, _) = *sources.choose_weighted(&mut rng, |(_, x)| *x).unwrap();
                let ingress_n = self.topo.topo_id(ingress);
                let egress = *self.egresses.choose(&mut rng).unwrap();
                let egress_n = self.topo.topo_id(egress);
                let mut path_topo = self.topo.clone();
                path_topo.map_link_weights(|_| rng.gen_range(1..=10));
                let Some((_, path)) = petgraph::algo::astar::astar(
                    &path_topo.graph,
                    ingress_n,
                    |f| f == egress_n,
                    |e| *e.weight(),
                    |_| 0,
                ) else {
                    continue;
                };
                let path = path.into_iter().map(|n| self.topo.net_id(n)).collect_vec();
                te_paths.push((ingress, egress, vec![path]));
                break;
            }
        }
        te_paths
    }
}

impl<P: Prefix, Rep, Att, Fri> GravityTMSampler<P, Rep, Att, Fri>
where
    Rep: Distribution<f64>,
    Att: Distribution<f64>,
    Fri: Distribution<f64>,
{
    /// If set to `true`, treat the friction as multiplicative: `rep(s) * att(d) * fri(s, d)`. By
    /// default (with `false`), it will treat it as a division: `rep(s) * att(d) / fri(s, d)`.
    pub fn invert_friction(&mut self, invert_friction: bool) -> &mut Self {
        self.invert_friction = invert_friction;
        self
    }

    /// If set to `true`, rescale the traffic matrix after sampling to restore the attraction
    /// distribution (which will be messed up when using a friction that is heavily tailed.)
    pub fn renormalize_attraction(&mut self, renormalize_attraction: bool) -> &mut Self {
        self.renormalize_attraction = renormalize_attraction;
        self
    }

    /// Set the distribution for the repulsive factor associated with "leaving" a source router.
    pub fn repulsion(&mut self, repulsion: Rep) -> &mut Self {
        self.repulsion = Some(repulsion);
        self
    }

    /// Set the distribution for the attractive factor, associating with "approaching" a destination
    /// prefix.
    pub fn attraction(&mut self, attraction: Att) -> &mut Self {
        self.attraction = Some(attraction);
        self
    }

    /// Set the distribution for the friction factor from a source router to a destination prefix.
    pub fn friction(&mut self, friction: Fri) -> &mut Self {
        self.friction = Some(friction);
        self
    }
}

/// Generate a traffic matrix by sampling data randomly.
#[derive(Debug)]
pub struct IidTMSampler<D> {
    /// The stored topology for sampling TE paths
    topo: Topology,
    /// Number of prefixes to include in the matrix. (10 by default)
    num_prefixes: u32,
    /// Routers in the network
    sources: Vec<RouterId>,
    /// Border routers in the network
    egresses: Vec<RouterId>,
    /// External routers in the network
    externals: Vec<RouterId>,
    /// Seed to use
    seed: Option<u64>,
    /// Distribution to sample from
    distr: Option<D>,
}

impl<P: Prefix, D: Distribution<f64>> TMSampler<P> for IidTMSampler<D> {
    /// Create a new, empty TrafficMatrixSampler
    fn new<Q>(net: &Network<P, Q>) -> Self {
        Self {
            topo: Topology::new(net),
            num_prefixes: 10,
            sources: net.internal_indices().collect(),
            egresses: net.ospf_network().external_edges().map(|e| e.int).unique().collect(),
            externals: net.external_indices().collect(),
            seed: None,
            distr: None,
        }
    }

    /// Set the seed to get a predictable outcome.
    fn prefixes(&mut self, num: u32) -> &mut Self {
        self.num_prefixes = num;
        self
    }

    /// Set the seed to get a predictable outcome.
    fn seed(&mut self, seed: u64) -> &mut Self {
        self.seed = Some(seed);
        self
    }

    /// Sample a traffic matrix
    fn sample(&mut self, progress: bool) -> TrafficMatrix<P> {
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        let distr = self.distr.as_ref().expect("dist is not set!");

        let mut result = HashMap::new();
        for p in (0..self.num_prefixes).my_progress("Sampling the traffic matrix", true, progress) {
            let dst = P::from(p);
            for src in &self.sources {
                let tc = TrafficClass { src: *src, dst };
                let demand = distr.sample(&mut rng);
                result.insert(tc, demand);
            }
        }
        result
    }

    fn current_state(
        &mut self,
        num_states: Option<usize>,
        progress: bool,
    ) -> HashMap<P, Vec<RouterId>> {
        if let Some(num_states) = num_states {
            random_distinct_current_state(
                &self.externals,
                (0..self.num_prefixes).map(P::from),
                num_states,
                self.seed,
                progress,
            )
        } else {
            random_current_state(
                &self.externals,
                (0..self.num_prefixes).map(P::from),
                self.seed,
                progress,
            )
        }
    }

    fn te_paths(&mut self, num_paths: usize) -> Vec<(RouterId, RouterId, Vec<Vec<RouterId>>)> {
        let mut rng = if let Some(seed) = self.seed {
            SmallRng::seed_from_u64(seed)
        } else {
            SmallRng::from_entropy()
        };
        let mut te_paths: Vec<(RouterId, RouterId, Vec<Vec<RouterId>>)> = Vec::new();
        for _ in 0..num_paths {
            loop {
                let ingress = *self.sources.choose(&mut rng).unwrap();
                let ingress_n = self.topo.topo_id(ingress);
                let egress = *self.egresses.choose(&mut rng).unwrap();
                let egress_n = self.topo.topo_id(egress);
                let mut path_topo = self.topo.clone();
                path_topo.map_link_weights(|_| rng.gen_range(1..=10));
                let Some((_, path)) = petgraph::algo::astar::astar(
                    &path_topo.graph,
                    ingress_n,
                    |f| f == egress_n,
                    |e| *e.weight(),
                    |_| 0,
                ) else {
                    continue;
                };
                let path = path.into_iter().map(|n| self.topo.net_id(n)).collect_vec();
                te_paths.push((ingress, egress, vec![path]));
                break;
            }
        }
        te_paths
    }
}

impl<D: Distribution<f64>> IidTMSampler<D> {
    /// Set an arbitrary distribution.
    pub fn distr(&mut self, distr: D) -> &mut Self {
        self.distr = Some(distr);
        self
    }
}

/// Uniform distribution
pub fn uniform(low: f64, high: f64) -> Uniform<f64> {
    Uniform::new(low, high)
}

/// Normal distribution
pub fn normal(mean: f64, std_dev: f64) -> Normal<f64> {
    Normal::new(mean, std_dev).unwrap()
}

/// Exponential distribution
pub fn exp(lambda: f64) -> Exp<f64> {
    Exp::new(lambda).expect("invalid lambda")
}

/// Poisson distribution
pub fn poisson(lambda: f64) -> Poisson<f64> {
    Poisson::new(lambda).expect("invalid lambda")
}

/// Beta distribution
pub fn beta(alpha: f64, beta: f64) -> Beta<f64> {
    Beta::new(alpha, beta).expect("invalid arguments")
}

/// Generate a LogNormal distribution according to shape `cv` with `mu = 0.0`.
pub fn log_normal(sigma: f64) -> LogNormal<f64> {
    LogNormal::new(0.0, sigma).expect("invalid arguments")
}

/// Generate a LogNormal distribution according to shape `cv` with an expected value of 1.0. A
/// larger shape parameter `cv` will create a heavier tail. `cv` is the Coefficient of Variation.
pub fn log_normal_cv(cv: f64) -> LogNormal<f64> {
    LogNormal::from_mean_cv(1.0, cv).expect("invalid arguments")
}

/// Generate a pareto distribution according to shape `alpha` with an expected value of 1.0. A
/// smaller shape parameter `alpha` will create a heavier tail.
pub fn pareto(alpha: f64) -> Offset<Pareto<f64>> {
    let exp = 1.0f64;
    let xm = exp / f64::powf(2.0, 1.0 / alpha);
    Offset::new(Pareto::new(xm, alpha).expect("invalid alpha"), -xm)
}

/// Offset a given distribution
#[derive(Debug)]
pub struct Offset<D> {
    d: D,
    offset: f64,
}

impl<D> Offset<D> {
    /// Create a new distribution that is equal to `d` with an offset.
    pub fn new(d: D, offset: f64) -> Self {
        Self { d, offset }
    }
}

impl<D: Distribution<f64>> Distribution<f64> for Offset<D> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.d.sample(rng) + self.offset
    }
}

/// Sample a random current state
fn random_current_state<P: Prefix>(
    externals: &[RouterId],
    prefixes: impl ExactSizeIterator<Item = P>,
    seed: Option<u64>,
    progress: bool,
) -> HashMap<P, Vec<RouterId>> {
    let mut rng = if let Some(seed) = seed {
        SmallRng::seed_from_u64(seed)
    } else {
        SmallRng::from_entropy()
    };

    let mut externals = externals.to_vec();

    prefixes
        .into_iter()
        .my_progress("Sampling the current state", false, progress)
        .map(|p| {
            externals.shuffle(&mut rng);
            let x = rng.gen_range(0.0..1.0);
            let n = if x < 0.6 {
                1
            } else if x < 0.9 {
                2
            } else {
                3
            };
            (p, externals.iter().take(n).copied().sorted().collect())
        })
        .collect()
}

/// Sample a random current state with a specified number of distinct states. The function will
/// first generate the states, and assign each prefix one of these states randomly (but using a
/// funciton that weightens some states more than others).
fn random_distinct_current_state<P: Prefix>(
    externals: &[RouterId],
    prefixes: impl ExactSizeIterator<Item = P>,
    num_states: usize,
    seed: Option<u64>,
    progress: bool,
) -> HashMap<P, Vec<RouterId>> {
    let p = match num_states {
        10 => 0.7,
        20 => 0.45,
        30 => 0.31,
        40 => 0.24,
        50 => 0.19,
        60 => 0.16,
        70 => 0.14,
        80 => 0.12,
        90 => 0.105,
        100 => 0.095,
        _ => panic!(),
    };
    let distr = rand_distr::Geometric::new(p).unwrap();

    let mut rng = if let Some(seed) = seed {
        SmallRng::seed_from_u64(seed)
    } else {
        SmallRng::from_entropy()
    };

    let externals = externals.to_vec();
    let mut initial_states: HashMap<u64, Vec<RouterId>> = HashMap::new();

    let mut sample = || {
        let n = distr.sample(&mut rng);
        initial_states
            .entry(n)
            .or_insert_with(|| {
                let x = rng.gen_range(0.0..1.0);
                let n = if x < 0.3 {
                    1
                } else if x < 0.6 {
                    2
                } else if x < 0.8 {
                    3
                } else {
                    4
                };
                externals.choose_multiple(&mut rng, n).copied().collect_vec()
            })
            .clone()
    };

    prefixes
        .into_iter()
        .my_progress("Sampling the current state", false, progress)
        .map(|p| (p, sample()))
        .collect()
}
