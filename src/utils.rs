//! Utility functions for the evaluation

use std::{cmp::Reverse, collections::HashMap};

use bgpsim::prelude::*;
use itertools::Itertools;
use maplit::hashmap;
use numpy::PyArray1;
use ordered_float::NotNan;
use pyo3::{
    types::{PyAnyMethods, PyDict},
    Python,
};

use crate::{
    analysis::PerformanceReport, performance::TrafficClass, traffic_matrix::TrafficMatrix,
};

/// Statistics of the traffic matrix
#[derive(Debug)]
pub struct TmStatistics<P> {
    /// The total demand
    pub total_demand: f64,
    /// The demand destined towards each prefix, sorted by the largest destination first.
    pub per_prefix: Vec<(P, f64)>,
    /// The demand sourced from each router, sorted by the largest router first
    pub per_source: Vec<(RouterId, f64)>,
}

impl<P: Prefix> TmStatistics<P> {
    /// Create a new TmStatistics from a traffic matrix.
    pub fn from_tm(tm: &TrafficMatrix<P>) -> Self {
        let mut per_source: HashMap<RouterId, f64> = HashMap::new();
        let mut per_prefix: HashMap<P, f64> = HashMap::new();
        let mut total_demand = 0.0;

        for (tc, demand) in tm {
            *per_source.entry(tc.src).or_default() += demand;
            *per_prefix.entry(tc.dst).or_default() += demand;
            total_demand += demand;
        }

        let mut per_source = per_source.into_iter().collect::<Vec<_>>();
        let mut per_prefix = per_prefix.into_iter().collect::<Vec<_>>();
        per_source.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        per_prefix.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        Self {
            total_demand,
            per_prefix,
            per_source,
        }
    }

    /// Create a new TM Statistics from prepared data.
    pub fn new(
        per_prefix: HashMap<P, f64>,
        per_source: HashMap<RouterId, f64>,
        norm_factor: f64,
    ) -> Self {
        // print the statistics
        let mut per_prefix =
            per_prefix.into_iter().map(|(n, x)| (n, x * norm_factor)).collect_vec();
        per_prefix.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        let mut per_source =
            per_source.into_iter().map(|(n, x)| (n, x * norm_factor)).collect_vec();
        per_source.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        let total_demand = per_source.iter().map(|(_, x)| x).sum::<f64>();

        Self {
            total_demand,
            per_prefix,
            per_source,
        }
    }

    /// Print the traffic matrix statistics.
    pub fn print_stats<Q, Ospf: OspfImpl>(&self, net: &Network<P, Q, Ospf>)
    where
        P: Prefix,
    {
        println!("Total demand:      {:.3}", self.total_demand);
        for k in [1, 10, 100, 1000] {
            println!(
                "[prefix] top {k: >4}: {:.3}",
                self.per_prefix.iter().map(|(_, x)| x).take(k).sum::<f64>()
            );
        }
        for (source, demand) in self.per_source.iter().take(10) {
            println!("[source] {: >8}: {demand:.3}", source.fmt(net));
        }
        println!();
    }
}

/// Difference of two performance reports, to measure the accuracy. This measures the difference of
/// the maximum link load of all links in the network. Make sure that the PerformanceReport is
/// generated based on the same data (both network, trafific matrix, and configuration).
#[derive(Debug, Clone)]
pub struct Difference {
    /// The minimum difference
    pub min: f64,
    /// The median difference
    pub median: f64,
    /// The maximum difference
    pub max: f64,
    /// The sum of all differences over all links.
    pub sum: f64,
    /// The average difference
    pub mean: f64,
}

/// Compute the difference of two performance reports. This measures the difference of the maximum
/// link load of all links in the network. Make sure that the PerformanceReport is generated based
/// on the same data (both network, trafific matrix, and configuration).
pub fn get_difference<P: Prefix>(
    a: &PerformanceReport<P>,
    b: &PerformanceReport<P>,
    norm: f64,
) -> Difference {
    let a = &a.loads;
    let b = &b.loads;

    let keys = a.keys().chain(b.keys()).collect::<std::collections::HashSet<_>>();
    let norm = 1.0 / norm;

    let mut data = keys
        .into_iter()
        .map(|k| a[k] - b[k])
        .map(f64::abs)
        .map(|x| x * norm)
        .collect::<Vec<_>>();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = data.first().copied().unwrap_or_default();
    let max = data.last().copied().unwrap_or_default();
    let median = data.get(data.len() / 2).copied().unwrap_or_default();
    let len = data.len();
    let sum = data.into_iter().sum::<f64>();
    let mean = sum / len as f64;
    Difference {
        min,
        median,
        max,
        sum,
        mean,
    }
}

/// Compute the number of heavy hitters that would suffice to reach the same error bounds.
pub fn compute_equivalend_top_k<P: Prefix>(
    total_traffic: f64,
    abs_error_bounds: f64,
    tm: &TrafficMatrix<P>,
) -> usize {
    let target_traffic = total_traffic - abs_error_bounds;
    let mut destination_traffic = HashMap::<P, NotNan<f64>>::new();
    for (tc, &bytes) in tm {
        *destination_traffic.entry(tc.dst).or_default() += NotNan::new(bytes).unwrap_or_default();
    }
    // sort the destinations by decreasing traffic
    let mut traffic_distr = destination_traffic.into_values().map(Reverse).collect::<Vec<_>>();
    let num_dst = traffic_distr.len();
    traffic_distr.sort();
    traffic_distr
        .into_iter()
        .scan((0usize, 0.0), |(i, sum), x| {
            *i += 1;
            *sum += x.0.into_inner();
            Some((*i, *sum))
        })
        .find(|(_, sum)| *sum > target_traffic)
        .map(|(i, _)| i)
        .unwrap_or(num_dst)
}

/// Lookup the parameters to use for sampling, such that the resulting matrix has the wanted
/// parameters.
#[allow(clippy::approx_constant)]
pub fn lookup_parameters(want: FittedParameters) -> Option<FittedParameters> {
    let lut = hashmap! {
        (30, 30, 25) => FittedParameters { attraction: 2.82, repulsion: 3.0, friction: 2.28 },
        (30, 30, 30) => FittedParameters { attraction: 2.76, repulsion: 3.0, friction: 2.74 },
        (30, 20, 40) => FittedParameters { attraction: 2.68, repulsion: 1.96, friction: 3.74 },
        (30, 30, 35) => FittedParameters { attraction: 2.66, repulsion: 3.0, friction: 3.2 },
        (30, 25, 40) => FittedParameters { attraction: 2.62, repulsion: 2.5, friction: 3.72 },
        (40, 30, 40) => FittedParameters { attraction: 3.7, repulsion: 2.98, friction: 3.62 },
        (45, 30, 40) => FittedParameters { attraction: 4.24, repulsion: 2.94, friction: 3.6 },
        (35, 30, 40) => FittedParameters { attraction: 3.14, repulsion: 3.0, friction: 3.64 },
        (30, 30, 40) => FittedParameters { attraction: 2.56, repulsion: 3.0, friction: 3.66 },
        (30, 30, 40) => FittedParameters { attraction: 2.56, repulsion: 3.0, friction: 3.66 },
        (25, 30, 40) => FittedParameters { attraction: 1.94, repulsion: 3.0, friction: 3.68 },
        (30, 35, 40) => FittedParameters { attraction: 2.48, repulsion: 3.5, friction: 3.62 },
        (30, 30, 45) => FittedParameters { attraction: 2.44, repulsion: 2.98, friction: 4.12 },
        (30, 30, 20) => FittedParameters { attraction: 2.9, repulsion: 3.0, friction: 1.82 },
        (30, 45, 40) => FittedParameters { attraction: 2.32, repulsion: 4.5, friction: 3.52 },
        (30, 30, 50) => FittedParameters { attraction: 2.3, repulsion: 2.94, friction: 4.6 },
        (30, 30, 40) => FittedParameters { attraction: 2.56, repulsion: 3.0, friction: 3.66 },
        (30, 40, 40) => FittedParameters { attraction: 2.4, repulsion: 4.0, friction: 3.56 },
        (20, 30, 40) => FittedParameters { attraction: 1.2, repulsion: 3.0, friction: 3.7 },
    };

    let want = (
        (want.attraction * 10.0).round() as i32,
        (want.repulsion * 10.0).round() as i32,
        (want.friction * 10.0).round() as i32,
    );
    lut.get(&want).cloned()
}

/// The fitted parameters of the three distributions underlying in the traffic matrix. Each of the
/// distributions is assumed to be distributed according to a LogNormal distributoin, and has the
/// parameter `mu = 0`.
#[derive(Debug, Clone)]
pub struct FittedParameters {
    /// Parameter of the attraction distribution.
    pub attraction: f64,
    /// Parameter of the repulsion distribution.
    pub repulsion: f64,
    /// Parameter of the (inverse) friction distribution. The friction is assumed to be a
    /// multiplicative factor, rather than a division (as it is originally defined in the gravity
    /// model.)
    pub friction: f64,
}

/// Fit the given traffic matrix by assuming it is sampled using a Gravity Model, with both
/// attraction, repulsion, and (inverse) friction following a LogNormal distribution.
///
/// Warning: This function requires python and `scipy` to be installed!
pub fn fit_parameters<P: Prefix>(tm: &TrafficMatrix<P>) -> FittedParameters {
    let mut per_src: HashMap<RouterId, f64> = HashMap::new();
    let mut per_dst: HashMap<P, f64> = HashMap::new();
    let total = tm.values().copied().sum::<f64>();

    let mut min_demand: f64 = 1.0;
    for (tc, demand) in tm {
        let rel_demand = demand / total;
        if rel_demand > 0.0 && rel_demand < min_demand {
            min_demand = rel_demand;
        }
        *per_src.entry(tc.src).or_default() += rel_demand;
        *per_dst.entry(tc.dst).or_default() += rel_demand;
    }

    let inv_friction = per_src
        .iter()
        .flat_map(|(&src, &src_p)| {
            per_dst.iter().map(move |(&dst, &dst_p)| {
                let demand = tm.get(&TrafficClass { src, dst }).copied().unwrap_or_default();
                let rel_demand = demand / total;
                let rel_demand = f64::max(rel_demand, min_demand);
                rel_demand / (src_p * dst_p)
            })
        })
        .collect::<Vec<_>>();
    let attraction = per_dst.into_values().collect::<Vec<_>>();
    let repulsion = per_src.into_values().collect::<Vec<_>>();

    FittedParameters {
        attraction: fit_sigma(attraction),
        repulsion: fit_sigma(repulsion),
        friction: fit_sigma(inv_friction),
    }
}

fn fit_sigma(x: Vec<f64>) -> f64 {
    let mut sigma = 0.0;
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let list = PyArray1::from_vec(py, x);
        let stats = py.import("scipy.stats").unwrap();
        let lognorm = stats.getattr("lognorm").unwrap();
        let fit = lognorm.getattr("fit").unwrap();
        let kargs = PyDict::new(py);
        kargs.set_item("floc", 0).unwrap();
        let fitted = fit.call((list,), Some(&kargs)).unwrap();
        let sigma_py = fitted.get_item(0).unwrap();
        sigma = sigma_py.extract().unwrap();
    });
    sigma
}

#[test]
fn test_fit_sigma() {
    approx::assert_relative_eq!(fit_sigma(vec![1.0, 1.0, 2.0]), 0.326, epsilon = 0.1);
}
