//! Velo: VErifying maximum link LOads in a changing world

#![deny(missing_debug_implementations)]
use std::{collections::HashMap, time::Instant};

use bgpsim::{
    network::Network,
    topology_zoo::TopologyZoo,
    types::{Prefix, RouterId},
};
use indicatif::{MultiProgress, ProgressBar, ProgressIterator};
use ordered_float::NotNan;
use serde::Serialize;
use time::{format_description::well_known::Rfc3339, OffsetDateTime};
use velo::{
    analysis::{ClusterSettings, Velo},
    explorer::{NetParams, NetworkSampler},
    progress_style,
    traffic_matrix::*,
    utils::{compute_equivalend_top_k, fit_parameters, get_difference, FittedParameters},
};

#[derive(Debug, Clone, Serialize)]
struct Datapoint {
    /// The topology
    topo: TopologyZoo,
    /// Number of nodes (internal) in the topology
    num_nodes: usize,
    /// Number of edges in the topology
    num_edges: usize,
    /// Number of external routers
    num_externals: usize,
    /// Random seed used to generate the config.
    config_seed: u64,
    /// Seed used to sample the traffic matrix
    tm_seed: u64,
    /// Number of destination prefixes
    num_prefixes: u32,
    repulsion_measured: f64,
    repulsion_target: f64,
    repulsion_sampled: f64,
    attraction_measured: f64,
    attraction_target: f64,
    attraction_sampled: f64,
    friction_measured: f64,
    friction_target: f64,
    friction_sampled: f64,
    /// Number of distinct initial states
    num_initial_states: usize,
    /// Clustering target
    cluster_target: usize,
    /// The cluster mode that is used by the KMeans algorithm
    cluster_mode: ClusterMode,
    /// Number of prefixes that are allowed to change
    input_changes: String,
    /// Number of traffic engineering paths
    num_te_paths: usize,
    /// Time to perform analysis on original (unclustered) data
    exact_time_sec: f64,
    /// Time to perform analysis on the approximated TM
    approx_time_sec: f64,
    /// Clustering time
    clustering_time_sec: f64,
    /// Clustering error relative to the total traffic.
    clustering_error: f64,
    /// Total demand in the traffic matrix
    total_demand: f64,
    /// Number of effective clusters
    num_clusters: usize,
    /// Minimum approximation error for the worst-case analysis, relative to the total traffic. The
    /// approximation error is the difference between the exact and the approximated result.
    approx_error_min: f64,
    /// Maximum approximation error for the worst-case analysis, relative to the total traffic. The
    /// approximation error is the difference between the exact and the approximated result.
    approx_error_max: f64,
    /// Mean approximation error for the worst-case analysis, relative to the total traffic. The
    /// approximation error is the difference between the exact and the approximated result.
    approx_error_mean: f64,
    /// Median approximation error for the worst-case analysis, relative to the total traffic. The
    /// approximation error is the difference between the exact and the approximated result.
    approx_error_median: f64,
    /// Total approximation error for the worst-case analysis, relative to the total traffic. The
    /// approximation error is the difference between the exact and the approximated result.
    approx_error_sum: f64,
    /// Speedup in efficiency compared to the top-k approach.
    efficiency_speedup: f64,
    /// Number of edges that cannot accurately reflect the variability in the worst-case.
    num_edges_inaccurate_variability: usize,
}

fn main() {
    let exploration = Exploration {
        topos: vec![
            TopologyZoo::TataNld,
            TopologyZoo::GtsCe,
            TopologyZoo::Colt,
            TopologyZoo::UsCarrier,
            TopologyZoo::DialtelecomCz,
            TopologyZoo::Cogentco,
        ],
        seeds: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        num_externals: Dimension::new(30, vec![]),
        num_prefixes: Dimension::new(100_000, vec![]),
        attraction: Dimension::new_f64(3.0, vec![2.0, 2.5, 3.5, 4.0]),
        repulsion: Dimension::new_f64(3.0, vec![2.0, 2.5, 3.5, 4.0]),
        friction: Dimension::new_f64(4.0, vec![2.0, 2.5, 3.0, 3.5, 4.5, 5.0]),
        cluster_target: Dimension::new(300, vec![100, 300, 600, 1000]),
        num_initial_states: Dimension::new(30, vec![]),
        cluster_mode: Dimension::new(ClusterMode::NormalizedScaled, vec![]),
        input_changes: Dimension::new(Some(100), vec![]),
    };

    let scenarios = exploration.scenarios();

    std::fs::create_dir_all("measurements").unwrap();
    let filename = format!(
        "measurements/accuracy-{}.csv",
        OffsetDateTime::now_utc().format(&Rfc3339).unwrap()
    );
    let mut writer = csv::WriterBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .from_path(&filename)
        .unwrap();

    let multi_pb = MultiProgress::new();
    let main_pb = multi_pb.insert(
        0,
        ProgressBar::new(scenarios.len() as u64)
            .with_style(progress_style())
            .with_message("Measuring accuracy..."),
    );

    for Parameters {
        net: net_params,
        tm: tm_params,
        cluster_target,
        cluster_mode,
        input_changes,
    } in scenarios.into_iter().progress_with(main_pb)
    {
        let net = net_params.sample().unwrap();
        let (traffic_matrix, current_state) = tm_params.sample(&net);

        let start_time = Instant::now();
        let mut config = Velo::new(&net);
        config.with_clustering(ClusterSettings::FixedNum {
            num: cluster_target,
            mode: cluster_mode,
        });
        config.multi_progress(multi_pb.clone());
        let velo = config.prepare(&current_state, &traffic_matrix);
        let clustering_time_sec = start_time.elapsed().as_secs_f64();

        let mut config_exact = Velo::new(&net);
        config_exact.multi_progress(multi_pb.clone());
        let velo_exact = config_exact.prepare(&current_state, &traffic_matrix);

        let start_time = Instant::now();
        let result = velo.analyze(input_changes, 0, false);
        let approx_time_sec = start_time.elapsed().as_secs_f64();

        let start_time = Instant::now();
        let result_exact = velo_exact.analyze(input_changes, 0, false);
        let exact_time_sec = start_time.elapsed().as_secs_f64();

        let total_demand = velo.total_demand();
        let approx_error = get_difference(&result_exact, &result, total_demand);

        let num_clusters_for_top_k =
            compute_equivalend_top_k(total_demand, result.pos_bounds, &traffic_matrix);
        let efficiency_speedup = num_clusters_for_top_k as f64 / velo.num_clusters() as f64;

        // approximate the traffic matrix parameters
        let FittedParameters {
            repulsion,
            attraction,
            friction,
        } = fit_parameters(&traffic_matrix);
        let FittedParameters {
            repulsion: repulsion_sampled,
            attraction: attraction_sampled,
            friction: friction_sampled,
        } = tm_params.sampling_parameters();

        let datapoint = Datapoint {
            topo: net_params.topo,
            num_nodes: net.internal_indices().count(),
            num_edges: net.ospf_network().internal_edges().count(),
            num_externals: net_params.num_externals,
            config_seed: net_params.config_seed,
            tm_seed: tm_params.tm_seed,
            num_prefixes: tm_params.num_prefixes,
            repulsion_sampled,
            repulsion_target: tm_params.repulsion.into_inner(),
            repulsion_measured: repulsion,
            attraction_sampled,
            attraction_target: tm_params.attraction.into_inner(),
            attraction_measured: attraction,
            friction_sampled,
            friction_target: tm_params.friction.into_inner(),
            friction_measured: friction,
            num_initial_states: tm_params.num_initial_states,
            cluster_target,
            cluster_mode,
            input_changes: input_changes.map(|x| x.to_string()).unwrap_or("inf".to_string()),
            num_te_paths: 0,
            clustering_time_sec,
            exact_time_sec,
            approx_time_sec,
            clustering_error: result.pos_bounds / total_demand,
            total_demand,
            num_clusters: velo.num_clusters(),
            approx_error_min: approx_error.min,
            approx_error_max: approx_error.max,
            approx_error_mean: approx_error.mean,
            approx_error_median: approx_error.median,
            approx_error_sum: approx_error.sum,
            efficiency_speedup,
            num_edges_inaccurate_variability: result.check_variability(&velo, false),
        };

        writer.serialize(&datapoint).unwrap();
        writer.flush().unwrap();
    }
}

#[derive(Debug, Clone)]
struct Dimension<T> {
    default: T,
    range: Vec<T>,
}

impl<T> Dimension<T> {
    pub fn new(default: T, range: Vec<T>) -> Self {
        Self { default, range }
    }
}

impl Dimension<NotNan<f64>> {
    pub fn new_f64(default: f64, range: Vec<f64>) -> Self {
        Self {
            default: NotNan::new(default).unwrap(),
            range: range.into_iter().map(|x| NotNan::new(x).unwrap()).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TmParams {
    /// Seed used to sample the traffic matrix
    tm_seed: u64,
    /// Number of destination prefixes
    num_prefixes: u32,
    /// The parameter for the repulsion factor. This is the sigma parameter for a log-normal
    /// distribution.
    repulsion: NotNan<f64>,
    /// The parameter for the friction factor. This is the sigma parameter of a log-normal
    /// distribution, and describes the shape.
    attraction: NotNan<f64>,
    /// The parameter for the friction factor. This is the sigma parameter of a log-normal
    /// distribution.
    friction: NotNan<f64>,
    /// Number of distinct initial states
    num_initial_states: usize,
}

impl TmParams {
    fn sample<P: Prefix, Q>(
        &self,
        net: &Network<P, Q>,
    ) -> (TrafficMatrix<P>, HashMap<P, Vec<RouterId>>) {
        let mut tm_sampler = GravityTMSampler::new(net);
        // lookup the parameters to use
        let FittedParameters {
            attraction,
            repulsion,
            friction,
        } = self.sampling_parameters();

        tm_sampler
            .prefixes(self.num_prefixes)
            .seed(self.tm_seed)
            .invert_friction(true)
            .renormalize_attraction(false)
            .attraction(log_normal(attraction))
            .repulsion(log_normal(repulsion))
            .friction(log_normal(friction));
        let tm = tm_sampler.sample(false);
        let current_state = tm_sampler.current_state(Some(self.num_initial_states), false);
        (tm, current_state)
    }

    fn sampling_parameters(&self) -> FittedParameters {
        velo::utils::lookup_parameters(FittedParameters {
            attraction: self.attraction.into_inner(),
            repulsion: self.repulsion.into_inner(),
            friction: self.friction.into_inner(),
        })
        .unwrap()
    }
}

#[derive(Debug, Clone)]
struct Parameters {
    /// Topology parameters
    net: NetParams,
    /// Traffic Matrix matameters
    tm: TmParams,
    /// Clustering target
    cluster_target: usize,
    /// The cluster mode that is used by the KMeans algorithm
    cluster_mode: ClusterMode,
    /// Number of prefixes that are allowed to change
    input_changes: Option<usize>,
}

struct Exploration {
    /// The topology
    topos: Vec<TopologyZoo>,
    /// Seed used to sample the traffic matrix
    seeds: Vec<u64>,
    /// Number of external routers
    num_externals: Dimension<usize>,
    /// Number of destination prefixes
    num_prefixes: Dimension<u32>,
    /// The parameter for the repulsion factor. This is the sigma parameter for a log-normal
    /// distribution.
    repulsion: Dimension<NotNan<f64>>,
    /// The parameter for the friction factor. This is the sigma parameter of a log-normal
    /// distribution.
    attraction: Dimension<NotNan<f64>>,
    /// The parameter for the friction factor. This is the sigma parameter of a log-normal
    /// distributiond.
    friction: Dimension<NotNan<f64>>,
    /// Number of distinct initial states
    num_initial_states: Dimension<usize>,
    /// Clustering target
    cluster_target: Dimension<usize>,
    /// The cluster mode that is used by the KMeans algorithm
    cluster_mode: Dimension<ClusterMode>,
    /// Number of prefixes that are allowed to change
    input_changes: Dimension<Option<usize>>,
}

impl Exploration {
    pub fn scenarios(&self) -> Vec<Parameters> {
        let num_externals = self.num_externals.default;
        let num_prefixes = self.num_prefixes.default;
        let repulsion_uniform_width = self.repulsion.default;
        let attraction = self.attraction.default;
        let friction_uniform_width = self.friction.default;
        let num_initial_states = self.num_initial_states.default;
        let cluster_target = self.cluster_target.default;
        let cluster_mode = self.cluster_mode.default;
        let input_changes = self.input_changes.default;

        let mut scenarios = Vec::new();

        for topo in self.topos.iter().copied() {
            for seed in self.seeds.iter().copied() {
                let (config_seed, tm_seed) = (seed, seed);

                for num_externals in self.num_externals.range.iter().copied() {
                    scenarios.push(Parameters {
                        net: NetParams {
                            topo,
                            num_externals,
                            config_seed,
                        },
                        tm: TmParams {
                            tm_seed,
                            num_prefixes,
                            repulsion: repulsion_uniform_width,
                            attraction,
                            friction: friction_uniform_width,
                            num_initial_states,
                        },
                        cluster_target,
                        cluster_mode,
                        input_changes,
                    })
                }
                for num_prefixes in self.num_prefixes.range.iter().copied() {
                    scenarios.push(Parameters {
                        net: NetParams {
                            topo,
                            num_externals,
                            config_seed,
                        },
                        tm: TmParams {
                            tm_seed,
                            num_prefixes,
                            repulsion: repulsion_uniform_width,
                            attraction,
                            friction: friction_uniform_width,
                            num_initial_states,
                        },
                        cluster_target,
                        cluster_mode,
                        input_changes,
                    })
                }
                for repulsion_uniform_width in self.repulsion.range.iter().copied() {
                    scenarios.push(Parameters {
                        net: NetParams {
                            topo,
                            num_externals,
                            config_seed,
                        },
                        tm: TmParams {
                            tm_seed,
                            num_prefixes,
                            repulsion: repulsion_uniform_width,
                            attraction,
                            friction: friction_uniform_width,
                            num_initial_states,
                        },
                        cluster_target,
                        cluster_mode,
                        input_changes,
                    })
                }
                for attraction in self.attraction.range.iter().copied() {
                    scenarios.push(Parameters {
                        net: NetParams {
                            topo,
                            num_externals,
                            config_seed,
                        },
                        tm: TmParams {
                            tm_seed,
                            num_prefixes,
                            repulsion: repulsion_uniform_width,
                            attraction,
                            friction: friction_uniform_width,
                            num_initial_states,
                        },
                        cluster_target,
                        cluster_mode,
                        input_changes,
                    })
                }
                for friction_uniform_width in self.friction.range.iter().copied() {
                    scenarios.push(Parameters {
                        net: NetParams {
                            topo,
                            num_externals,
                            config_seed,
                        },
                        tm: TmParams {
                            tm_seed,
                            num_prefixes,
                            repulsion: repulsion_uniform_width,
                            attraction,
                            friction: friction_uniform_width,
                            num_initial_states,
                        },
                        cluster_target,
                        cluster_mode,
                        input_changes,
                    })
                }
                for num_initial_states in self.num_initial_states.range.iter().copied() {
                    scenarios.push(Parameters {
                        net: NetParams {
                            topo,
                            num_externals,
                            config_seed,
                        },
                        tm: TmParams {
                            tm_seed,
                            num_prefixes,
                            repulsion: repulsion_uniform_width,
                            attraction,
                            friction: friction_uniform_width,
                            num_initial_states,
                        },
                        cluster_target,
                        cluster_mode,
                        input_changes,
                    })
                }
                for cluster_target in self.cluster_target.range.iter().copied() {
                    scenarios.push(Parameters {
                        net: NetParams {
                            topo,
                            num_externals,
                            config_seed,
                        },
                        tm: TmParams {
                            tm_seed,
                            num_prefixes,
                            repulsion: repulsion_uniform_width,
                            attraction,
                            friction: friction_uniform_width,
                            num_initial_states,
                        },
                        cluster_target,
                        cluster_mode,
                        input_changes,
                    })
                }
                for cluster_mode in self.cluster_mode.range.iter().copied() {
                    scenarios.push(Parameters {
                        net: NetParams {
                            topo,
                            num_externals,
                            config_seed,
                        },
                        tm: TmParams {
                            tm_seed,
                            num_prefixes,
                            repulsion: repulsion_uniform_width,
                            attraction,
                            friction: friction_uniform_width,
                            num_initial_states,
                        },
                        cluster_target,
                        cluster_mode,
                        input_changes,
                    })
                }
                for input_changes in self.input_changes.range.iter().copied() {
                    scenarios.push(Parameters {
                        net: NetParams {
                            topo,
                            num_externals,
                            config_seed,
                        },
                        tm: TmParams {
                            tm_seed,
                            num_prefixes,
                            repulsion: repulsion_uniform_width,
                            attraction,
                            friction: friction_uniform_width,
                            num_initial_states,
                        },
                        cluster_target,
                        cluster_mode,
                        input_changes,
                    })
                }
            }
        }

        scenarios
    }
}
