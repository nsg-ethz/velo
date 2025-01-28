use std::{
    io::{stdout, Write},
    ops::AddAssign,
};

use bgpsim::types::Prefix;
use ndarray::{s, Array1, Array2, ArrayView1};
use ordered_float::NotNan;
use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::MyProgressIterator;

use super::cluster::{ClusterMode, Clusters, TrafficMatrixData};

/// Common interface for defining a clustering
pub trait Clustering {
    /// Perform the clustering
    fn cluster<'a, P: Prefix, ID>(
        &self,
        data: &'a TrafficMatrixData<P, ID>,
        progress: bool,
    ) -> Clusters<'a, P>;

    /// The clustering mode to use when computing errors.
    fn mode(&self) -> ClusterMode;
}

/// Perform the clustering by choosing the `k` heavyest prefixes as their own clusters, and
/// combining all other prefix into their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HeaviestK {
    /// Number of clusters
    pub k: usize,
    /// The clustering mode to use
    pub mode: ClusterMode,
}

impl Clustering for HeaviestK {
    fn cluster<'a, P: Prefix, ID>(
        &self,
        data: &'a TrafficMatrixData<P, ID>,
        _progress: bool,
    ) -> Clusters<'a, P> {
        let k = self.k;
        let assignment: Vec<usize> = if k >= data.num_prefixes {
            (0..data.num_prefixes).collect()
        } else {
            // get the load for each prefix
            let mut prefixes_loads: Vec<(usize, f64)> =
                data.data.iter().enumerate().map(|(i, x)| (i, x.scale)).collect();
            // compute the rank by sorting them
            prefixes_loads.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            // store the rank
            let mut prefixes_ranks = prefixes_loads
                .into_iter()
                .enumerate()
                .map(|(rank, (i, _))| (i, rank))
                .collect::<Vec<_>>();
            // sort by the prefix id
            prefixes_ranks.sort_by(|(i, _), (j, _)| i.cmp(j));
            // extract the cluster
            prefixes_ranks.into_iter().map(|(_, r)| if r < k { r } else { k }).collect()
        };
        Clusters::from_prediction(data, &assignment)
    }

    fn mode(&self) -> ClusterMode {
        self.mode
    }
}

/// Perform the clustering by only choosing a single cluster point, and assigning all prefixes to
/// that cluster
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SingleCluster {
    /// How to compute the center point and the error.
    pub mode: ClusterMode,
}

impl Default for SingleCluster {
    fn default() -> Self {
        Self {
            mode: ClusterMode::NormalizedScaled,
        }
    }
}

impl Clustering for SingleCluster {
    fn cluster<'a, P: Prefix, ID>(
        &self,
        data: &'a TrafficMatrixData<P, ID>,
        _progress: bool,
    ) -> Clusters<'a, P> {
        Clusters::from_prediction(data, &vec![0; data.num_prefixes])
    }

    fn mode(&self) -> ClusterMode {
        self.mode
    }
}

/// How to initialize KMeans
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KMeansInit {
    /// Initialize by picking random points (that are within the range of values)
    #[default]
    Random,
    /// Initialize by picking random points (that are within the range of values) from a seeded value.
    RandomSeeded(u64),
    /// Initialize by picking the datapoints equal to the prefixes with the most demand.
    HeaviestPrefixes,
}

/// Perform KMeans clustering.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeans {
    /// Number of clusters, defaults to 50
    pub k: usize,
    /// Clustering mode that governs in which cluster a point should belong, and how to compute the
    /// cost of each cluster. Default value is `ClusterMode::NormalizedScaled`.
    pub mode: ClusterMode,
    /// How to initialize the clusters at the beginning. Default value is
    /// `KMeansInit::HeaviestPrefixes`.
    pub init: KMeansInit,
    /// Maximum number of iterations. Default value is 10000.
    pub max_iter: usize,
    /// Relative tolerance with regards to the difference in the cluister centers of two consecutive
    /// iterations to declare convergence. Default value is 1e-6.
    pub tolerance: f64,
    /// Whether to use parallel cores to perform clustering
    pub parallel: bool,
}

impl Default for KMeans {
    fn default() -> Self {
        Self {
            k: 50,
            mode: ClusterMode::NormalizedScaled,
            init: KMeansInit::HeaviestPrefixes,
            max_iter: 150,
            tolerance: 1e-5,
            parallel: true,
        }
    }
}

/// Initialized KMeans
#[derive(Debug, Clone, PartialEq)]
pub struct KMeansData {
    /// Number of clusters, defaults to 8
    pub k: usize,
    /// Clustering mode that governs in which cluster a point should belong, and how to compute the
    /// cost of each cluster. Default value is `ClusterMode::Direct`.
    pub mode: ClusterMode,
    /// How to initialize the clusters at the beginning. Default value is `KMeansInit::Random`.
    pub init: KMeansInit,
    /// Maximum number of iterations. Default value is 300.
    pub max_iter: usize,
    /// Relative tolerance with regards to the difference in the cluister centers of two consecutive
    /// iterations to declare convergence. Default value is 1e-5.
    pub tolerance: f64,
    /// Whether to use parallel cores to perform clustering
    pub parallel: bool,
    /// Number of dimensions of each datapoint
    n_dim: usize,
    /// Number of datapoints
    n_data: usize,
    /// scaling factor for each datapoint
    scale: Array1<f64>,
    /// data
    data: Array2<f64>,
}

impl Clustering for KMeans {
    fn cluster<'a, P: Prefix, ID>(
        &self,
        data: &'a TrafficMatrixData<P, ID>,
        progress: bool,
    ) -> Clusters<'a, P> {
        self.setup_data(data).cluster(data, progress)
    }

    fn mode(&self) -> ClusterMode {
        self.mode
    }
}

impl KMeans {
    fn setup_data<P: Prefix, ID>(&self, input: &TrafficMatrixData<P, ID>) -> KMeansData {
        let mut x = KMeansData {
            k: self.k,
            mode: self.mode,
            init: self.init,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            parallel: self.parallel,
            n_dim: input.num_sources,
            n_data: input.num_prefixes,
            scale: Array1::<f64>::zeros(input.num_prefixes),
            data: Array2::<f64>::zeros((input.num_prefixes, input.num_sources)),
        };

        for (i, m) in input.data.iter().enumerate() {
            x.scale[i] = m.scale;
            let data = if self.mode.normalized() {
                &m.norm_demand
            } else {
                &m.demand
            };
            for (j, demand) in data.iter().enumerate() {
                x.data[(i, j)] = *demand;
            }
        }

        x
    }
}

impl Clustering for KMeansData {
    fn cluster<'a, P: Prefix, ID>(
        &self,
        data: &'a TrafficMatrixData<P, ID>,
        progress: bool,
    ) -> Clusters<'a, P> {
        // we must have at least as many datapoints as clusters
        assert!(data.data.len() >= self.k);

        let min_delta = self.min_delta();
        let mut clusters = self.initialize();

        for _i in (0..self.max_iter).my_spinner(
            format!("Clustering KMeans (k={})", self.k),
            false,
            progress,
        ) {
            stdout().flush().unwrap();
            let assignment = self.nearest_clustering(&clusters);
            let new_clusters = self.find_clusters(&assignment);

            // check if the difference between the clusters is small
            let delta = new_clusters
                .iter()
                .zip(clusters.iter())
                .map(|(new, old)| (*new - *old).abs())
                .sum::<f64>();
            clusters = new_clusters;
            if delta < min_delta {
                // converged!
                break;
            }
        }

        // compute the clustering and return it.
        let prediction = self.nearest_clustering(&clusters);
        Clusters::from_prediction(data, prediction.as_slice().unwrap())
    }

    fn mode(&self) -> ClusterMode {
        self.mode
    }
}

impl KMeansData {
    /// Perform the clustering by assigning points to their nearest cluster.
    fn nearest_clustering(&self, clusters: &Array2<f64>) -> Array1<usize> {
        let map_op = |(x, &scale)| {
            // find the best assignment to the clusters
            clusters
                .rows()
                .into_iter()
                .map(|c| self.difference(x, c, scale))
                .enumerate()
                .min_by(|(_, a), (_, b)| a.cmp(b))
                .map(|(i, _)| i)
                .unwrap()
        };
        // for each datapoint, figure out which cluster it should belong to.
        let iter = self.data.rows().into_iter().zip(self.scale.iter());
        if self.parallel {
            iter.collect::<Vec<_>>()
                .into_par_iter()
                .map(map_op)
                .collect::<Vec<_>>()
                .into_iter()
                .collect()
        } else {
            iter.map(map_op).collect()
        }
    }

    /// Find the cluster points according to `self.mode`.
    fn find_clusters(&self, assignment: &Array1<usize>) -> Array2<f64> {
        let mut clustering = Array2::<f64>::zeros((self.k, self.n_dim));

        let map_op = |i_c| {
            let mut cluster_sum = Array1::<f64>::zeros(self.n_dim);
            let factor = assignment
                // iterate over all datapoints of that cluster
                .iter()
                .enumerate()
                // only take those datapoints that correspond to that cluster
                .filter(|(_, c)| **c == i_c)
                // get the data of those datapoints
                .map(|(i_x, _)| (self.data.slice(s![i_x, ..]), self.scale[i_x]))
                // add the data to the cluster_sum (potentially weightened if necessary)
                .map(|(x, scale)| match self.mode {
                    ClusterMode::Direct | ClusterMode::Normalized => {
                        cluster_sum.add_assign(&x);
                        1.0
                    }
                    ClusterMode::NormalizedScaled => {
                        let mut x = x.to_owned();
                        x *= scale;
                        cluster_sum.add_assign(&x);
                        scale
                    }
                })
                // create a larger array that contains all those datapoints
                .sum::<f64>();
            if factor != 0.0 {
                cluster_sum /= factor;
            }
            (i_c, cluster_sum)
        };

        let insert_op = |(i_c, cluster_sum)| {
            clustering.slice_mut(s![i_c, ..]).add_assign(&cluster_sum);
        };

        if self.parallel {
            (0..self.k)
                .into_par_iter()
                .map(map_op)
                .collect::<Vec<_>>()
                .into_iter()
                .for_each(insert_op);
        } else {
            (0..self.k).map(map_op).for_each(insert_op);
        }
        clustering
    }

    /// Compute the difference between two points
    fn difference(&self, x: ArrayView1<f64>, y: ArrayView1<f64>, scale: f64) -> NotNan<f64> {
        let diff = x
            .as_slice()
            .unwrap()
            .iter()
            .zip(y.as_slice().unwrap())
            .map(|(x, y)| (*x - *y).abs())
            .sum::<f64>();

        NotNan::new(match self.mode {
            ClusterMode::Direct | ClusterMode::Normalized => diff,
            ClusterMode::NormalizedScaled => diff * scale,
        })
        .unwrap()
    }

    /// Compute the minimum distance required to tell that the clustering has converged.
    fn min_delta(&self) -> f64 {
        match self.mode {
            ClusterMode::Direct => self.scale.sum() * self.tolerance,
            ClusterMode::Normalized | ClusterMode::NormalizedScaled => self.tolerance,
        }
    }

    /// Generate initial clustering
    fn initialize(&self) -> Array2<f64> {
        match self.init {
            KMeansInit::Random | KMeansInit::RandomSeeded(_) => self.initialize_random(),
            KMeansInit::HeaviestPrefixes => self.initialize_heaviest(),
        }
    }

    /// Generate initial clustering
    fn initialize_heaviest(&self) -> Array2<f64> {
        let mut prefixes_loads =
            self.scale.as_slice().unwrap().iter().copied().enumerate().collect::<Vec<_>>();

        prefixes_loads.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        let mut clusters = Array2::<f64>::zeros((self.k, self.n_dim));
        for (i_c, (i_x, _)) in prefixes_loads.into_iter().enumerate().take(self.k) {
            clusters.slice_mut(s![i_c, ..]).add_assign(&self.data.slice(s![i_x, ..]));
        }

        clusters
    }

    /// Generate initial clustering
    fn initialize_random(&self) -> Array2<f64> {
        let mut rng = if let KMeansInit::RandomSeeded(seed) = self.init {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        // pick k random samples to use as clusters
        let mut indices: Vec<_> = (0..self.n_data).collect();
        indices.shuffle(&mut rng);

        // create the clusters
        let mut clusters = Array2::<f64>::zeros((self.k, self.n_dim));
        for (i_c, i_x) in indices.into_iter().enumerate().take(self.k) {
            clusters.slice_mut(s![i_c, ..]).add_assign(&self.data.slice(s![i_x, ..]));
        }

        clusters
    }
}
