//! Velo: VErifying maximum link LOads in a changing world

#![deny(missing_docs, missing_debug_implementations)]
use std::{collections::HashMap, time::Instant};

use indicatif::{MultiProgress, ProgressBar, ProgressIterator};
use ordered_float::NotNan;
use rand::prelude::*;
use serde::Serialize;
use time::{format_description::well_known::Rfc3339, OffsetDateTime};
use velo::{
    algorithms::Topology,
    analysis::{ClusterSettings, Velo},
    explorer::{NetParams, NetworkSampler},
    progress_style,
    traffic_matrix::*,
};

use bgpsim::{
    network::Network,
    ospf::OspfImpl,
    topology_zoo::TopologyZoo,
    types::{Prefix, RouterId},
};
use itertools::Itertools;

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
    /// The parameter for the repulsion factor. This is a uniform distribution, and describes the
    /// (multiplicative) amount by which the upper limit is larger than the lower limit.
    repulsion_uniform_width: f64,
    /// The parameter for the friction factor. This is a pareto distribution, and describes the
    /// shape. If this number is negative, then the friction is not sampled with a pareto
    /// distribution.
    attraction_shape: f64,
    /// The parameter for the friction factor. This is a uniform distribution, and describes the
    /// (multiplicative) amount by which the upper limit is larger than the lower limit.
    friction_uniform_width: f64,
    /// Number of distinct initial states
    num_initial_states: usize,
    /// Clustering target
    cluster_target: usize,
    /// The cluster mode that is used by the KMeans algorithm
    cluster_mode: ClusterMode,
    /// Number of link failures
    link_failures: usize,
    /// Number of prefixes that are allowed to change
    input_changes: String,
    /// Number of traffic engineering paths
    num_te_paths: usize,
    /// The total time
    running_time_sec: f64,
    /// Analysis time, without clustering
    analysis_time_sec: f64,
    /// Clustering time
    clustering_time_sec: f64,
    /// Clustering error relative to the total traffic.
    clustering_error: f64,
    /// Total demand in the traffic matrix
    total_demand: f64,
    /// Number of effective clusters
    num_clusters: usize,
    /// Number of edges that cannot accurately reflect the variability in the worst-case.
    num_edges_inaccurate_variability: usize,
}

fn main() {
    let exploration = Exploration {
        seeds: vec![1, 2, 3, 4, 5],
        topo: Dimension::new(
            TopologyZoo::Cogentco,
            TopologyZoo::topologies_increasing_nodes()
                .iter()
                .copied()
                .filter(|t| t.num_internals() >= 40)
                .collect(),
        ),
        num_externals: Dimension::new(30, vec![100]),
        num_prefixes: Dimension::new(100_000, vec![]),
        repulsion_uniform_width: Dimension::new_f64(8.0, vec![]),
        attraction_shape: Dimension::new_f64(0.8, vec![]),
        friction_uniform_width: Dimension::new_f64(2.0, vec![]),
        num_initial_states: Dimension::new(30, vec![]),
        cluster_target: Dimension::new(300, vec![100, 600, 1000]),
        cluster_mode: Dimension::new(ClusterMode::NormalizedScaled, vec![]),
        link_failures: Dimension::new(2, vec![0, 1, 2]),
        input_changes: Dimension::new(Some(10), vec![Some(1), Some(100), None]),
        num_te_paths: Dimension::new(0, vec![1, 3, 5]),
    };

    let scenarios = exploration.scenarios();
    let mut cache: HashMap<(_, _), (_, _, _)> = HashMap::new();

    let mut rng = SmallRng::seed_from_u64(1);

    std::fs::create_dir_all("measurements").unwrap();
    let filename = format!(
        "measurements/running-time-{}.csv",
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
            .with_message("Measuring running time..."),
    );

    for Parameters {
        net: net_params,
        tm: tm_params,
        cluster_target,
        cluster_mode,
        link_failures,
        input_changes,
        num_te_paths,
    } in scenarios.into_iter().progress_with(main_pb)
    {
        let (net, traffic_matrix, current_state) =
            cache.entry((net_params, tm_params)).or_insert_with_key(|(topo, tm)| {
                let net = topo.sample().unwrap();
                let (tm, state) = tm.sample(&net);
                (net, tm, state)
            });

        let start_time = Instant::now();
        let mut config = Velo::new(net);
        config.with_clustering(ClusterSettings::FixedNum {
            num: cluster_target,
            mode: cluster_mode,
        });
        config.directional_link_failures();
        config.multi_progress(multi_pb.clone());

        // prepare the te paths
        let te_paths = (0..10).map(|_| sample_paths(net, num_te_paths, &mut rng)).collect_vec();
        let te_paths = te_paths.into_iter().map(|p| config.prepare_te_paths(p)).collect_vec();

        let mut velo = config.prepare(current_state, traffic_matrix);
        let clustering_time_sec = start_time.elapsed().as_secs_f64();

        // setup te paths
        for (i, te_paths) in te_paths.iter().enumerate() {
            velo.install_te_paths_at(i, te_paths)
        }

        let start_time = Instant::now();
        let result = velo.analyze(input_changes, link_failures, false);
        let analysis_time_sec = start_time.elapsed().as_secs_f64();
        let running_time_sec = clustering_time_sec + analysis_time_sec;

        let datapoint = Datapoint {
            topo: net_params.topo,
            num_nodes: net.internal_indices().count(),
            num_edges: net.ospf_network().internal_edges().count(),
            num_externals: net_params.num_externals,
            config_seed: net_params.config_seed,
            tm_seed: tm_params.tm_seed,
            num_prefixes: tm_params.num_prefixes,
            repulsion_uniform_width: tm_params.repulsion_uniform_width.into_inner(),
            attraction_shape: tm_params.attraction_shape.into_inner(),
            friction_uniform_width: tm_params.friction_uniform_width.into_inner(),
            num_initial_states: tm_params.num_initial_states,
            cluster_target,
            cluster_mode,
            link_failures,
            input_changes: input_changes.map(|x| x.to_string()).unwrap_or("inf".to_string()),
            num_te_paths,
            running_time_sec,
            analysis_time_sec,
            clustering_time_sec,
            clustering_error: result.pos_bounds,
            total_demand: velo.total_demand(),
            num_clusters: velo.num_clusters(),
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
    /// The parameter for the repulsion factor. This is a uniform distribution, and describes the
    /// (multiplicative) amount by which the upper limit is larger than the lower limit.
    repulsion_uniform_width: NotNan<f64>,
    /// The parameter for the friction factor. This is a pareto distribution, and describes the
    /// shape. If this number is negative, then the friction is not sampled with a pareto
    /// distribution.
    attraction_shape: NotNan<f64>,
    /// The parameter for the friction factor. This is a uniform distribution, and describes the
    /// (multiplicative) amount by which the upper limit is larger than the lower limit.
    friction_uniform_width: NotNan<f64>,
    /// Number of distinct initial states
    num_initial_states: usize,
}

impl TmParams {
    fn sample<P: Prefix, Q>(
        &self,
        net: &Network<P, Q>,
    ) -> (TrafficMatrix<P>, HashMap<P, Vec<RouterId>>) {
        let mut tm_sampler = GravityTMSampler::new(net);
        tm_sampler
            .prefixes(self.num_prefixes)
            .seed(self.tm_seed)
            .attraction(pareto(self.attraction_shape.into_inner()))
            .repulsion(uniform(1.0, self.repulsion_uniform_width.into_inner()))
            .friction(uniform(1.0, self.friction_uniform_width.into_inner()));
        let tm = tm_sampler.sample(false);
        let current_state = tm_sampler.current_state(Some(self.num_initial_states), false);
        (tm, current_state)
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
    /// Number of link failures
    link_failures: usize,
    /// Number of prefixes that are allowed to change
    input_changes: Option<usize>,
    /// Number of traffic engineering paths
    num_te_paths: usize,
}

struct Exploration {
    /// Seed used to sample both the config and the traffic matrix
    seeds: Vec<u64>,
    /// The topology
    topo: Dimension<TopologyZoo>,
    /// Number of external routers
    num_externals: Dimension<usize>,
    /// Number of destination prefixes
    num_prefixes: Dimension<u32>,
    /// The parameter for the repulsion factor. This is a uniform distribution, and describes the
    /// (multiplicative) amount by which the upper limit is larger than the lower limit.
    repulsion_uniform_width: Dimension<NotNan<f64>>,
    /// The parameter for the friction factor. This is a pareto distribution, and describes the
    /// shape. If this number is negative, then the friction is not sampled with a pareto
    /// distribution.
    attraction_shape: Dimension<NotNan<f64>>,
    /// The parameter for the friction factor. This is a uniform distribution, and describes the
    /// (multiplicative) amount by which the upper limit is larger than the lower limit.
    friction_uniform_width: Dimension<NotNan<f64>>,
    /// Number of distinct initial states
    num_initial_states: Dimension<usize>,
    /// Clustering target
    cluster_target: Dimension<usize>,
    /// The cluster mode that is used by the KMeans algorithm
    cluster_mode: Dimension<ClusterMode>,
    /// Number of prefixes that are allowed to change
    link_failures: Dimension<usize>,
    /// Number of prefixes that are allowed to change
    input_changes: Dimension<Option<usize>>,
    /// Number of traffic engineering paths
    num_te_paths: Dimension<usize>,
}

impl Exploration {
    pub fn scenarios(&self) -> Vec<Parameters> {
        let topo = self.topo.default;
        let num_externals = self.num_externals.default;
        let num_prefixes = self.num_prefixes.default;
        let repulsion_uniform_width = self.repulsion_uniform_width.default;
        let attraction_shape = self.attraction_shape.default;
        let friction_uniform_width = self.friction_uniform_width.default;
        let num_initial_states = self.num_initial_states.default;
        let cluster_target = self.cluster_target.default;
        let cluster_mode = self.cluster_mode.default;
        let link_failures = self.link_failures.default;
        let input_changes = self.input_changes.default;
        let num_te_paths = self.num_te_paths.default;

        let mut scenarios = Vec::new();

        for seed in self.seeds.iter().copied() {
            let (config_seed, tm_seed) = (seed, seed);

            for topo in self.topo.range.iter().copied() {
                for link_failures in self.link_failures.range.iter().copied() {
                    scenarios.push(Parameters {
                        net: NetParams {
                            topo,
                            num_externals,
                            config_seed,
                        },
                        tm: TmParams {
                            tm_seed,
                            num_prefixes,
                            repulsion_uniform_width,
                            attraction_shape,
                            friction_uniform_width,
                            num_initial_states,
                        },
                        cluster_target,
                        cluster_mode,
                        input_changes,
                        link_failures,
                        num_te_paths,
                    })
                }
            }

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
                        repulsion_uniform_width,
                        attraction_shape,
                        friction_uniform_width,
                        num_initial_states,
                    },
                    cluster_target,
                    cluster_mode,
                    input_changes,
                    link_failures,
                    num_te_paths,
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
                        repulsion_uniform_width,
                        attraction_shape,
                        friction_uniform_width,
                        num_initial_states,
                    },
                    cluster_target,
                    cluster_mode,
                    input_changes,
                    link_failures,
                    num_te_paths,
                })
            }
            for repulsion_uniform_width in self.repulsion_uniform_width.range.iter().copied() {
                scenarios.push(Parameters {
                    net: NetParams {
                        topo,
                        num_externals,
                        config_seed,
                    },
                    tm: TmParams {
                        tm_seed,
                        num_prefixes,
                        repulsion_uniform_width,
                        attraction_shape,
                        friction_uniform_width,
                        num_initial_states,
                    },
                    cluster_target,
                    cluster_mode,
                    input_changes,
                    link_failures,
                    num_te_paths,
                })
            }
            for attraction_shape in self.attraction_shape.range.iter().copied() {
                scenarios.push(Parameters {
                    net: NetParams {
                        topo,
                        num_externals,
                        config_seed,
                    },
                    tm: TmParams {
                        tm_seed,
                        num_prefixes,
                        repulsion_uniform_width,
                        attraction_shape,
                        friction_uniform_width,
                        num_initial_states,
                    },
                    cluster_target,
                    cluster_mode,
                    input_changes,
                    link_failures,
                    num_te_paths,
                })
            }
            for friction_uniform_width in self.friction_uniform_width.range.iter().copied() {
                scenarios.push(Parameters {
                    net: NetParams {
                        topo,
                        num_externals,
                        config_seed,
                    },
                    tm: TmParams {
                        tm_seed,
                        num_prefixes,
                        repulsion_uniform_width,
                        attraction_shape,
                        friction_uniform_width,
                        num_initial_states,
                    },
                    cluster_target,
                    cluster_mode,
                    input_changes,
                    link_failures,
                    num_te_paths,
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
                        repulsion_uniform_width,
                        attraction_shape,
                        friction_uniform_width,
                        num_initial_states,
                    },
                    cluster_target,
                    cluster_mode,
                    input_changes,
                    link_failures,
                    num_te_paths,
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
                        repulsion_uniform_width,
                        attraction_shape,
                        friction_uniform_width,
                        num_initial_states,
                    },
                    cluster_target,
                    cluster_mode,
                    input_changes,
                    link_failures,
                    num_te_paths,
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
                        repulsion_uniform_width,
                        attraction_shape,
                        friction_uniform_width,
                        num_initial_states,
                    },
                    cluster_target,
                    cluster_mode,
                    input_changes,
                    link_failures,
                    num_te_paths,
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
                        repulsion_uniform_width,
                        attraction_shape,
                        friction_uniform_width,
                        num_initial_states,
                    },
                    cluster_target,
                    cluster_mode,
                    input_changes,
                    link_failures,
                    num_te_paths,
                })
            }
            for num_te_paths in self.num_te_paths.range.iter().copied() {
                scenarios.push(Parameters {
                    net: NetParams {
                        topo,
                        num_externals,
                        config_seed,
                    },
                    tm: TmParams {
                        tm_seed,
                        num_prefixes,
                        repulsion_uniform_width,
                        attraction_shape,
                        friction_uniform_width,
                        num_initial_states,
                    },
                    cluster_target,
                    cluster_mode,
                    input_changes,
                    link_failures,
                    num_te_paths,
                })
            }
        }

        scenarios
    }
}

fn sample_paths<P: Prefix, Q, Ospf: OspfImpl>(
    net: &Network<P, Q, Ospf>,
    num: usize,
    rng: &mut SmallRng,
) -> Vec<(RouterId, RouterId, Vec<Vec<RouterId>>)> {
    let mut te_paths: Vec<(RouterId, RouterId, Vec<Vec<RouterId>>)> = Vec::new();

    let mut ingresses = net.internal_indices().collect_vec();
    let mut egresses = net.ospf_network().external_edges().map(|e| e.int).unique().collect_vec();
    egresses.shuffle(rng);
    let mut topo = Topology::new(net);

    let mut egress_counter = 0;

    while te_paths.len() < num {
        let egress = egresses[egress_counter];
        egress_counter = (egress_counter + 1) % egresses.len();
        ingresses.shuffle(rng);
        'inner: for ingress in ingresses.iter().copied() {
            let ingress_n = topo.topo_id(ingress);
            let egress_n = topo.topo_id(egress);
            let Some((_, path)) = petgraph::algo::astar::astar(
                &topo.graph,
                ingress_n,
                |f| f == egress_n,
                |e| *e.weight(),
                |_| 0,
            ) else {
                continue 'inner;
            };
            // push the new path
            let path = path.into_iter().map(|n| topo.net_id(n)).collect_vec();
            te_paths.push((ingress, egress, vec![path]));
            // re-shuffle the weights
            topo.map_link_weights(|_| rng.gen_range(1..=10));
            // break out of the inner loop
            break 'inner;
        }
    }
    te_paths
}
