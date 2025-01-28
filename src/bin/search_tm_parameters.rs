//! Velo: VErifying maximum link LOads in a changing world

#![deny(missing_docs, missing_debug_implementations)]
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use bgpsim::{
    event::BasicEventQueue,
    network::Network,
    topology_zoo::TopologyZoo,
    types::{Prefix, SimplePrefix},
};

use ordered_float::NotNan;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use serde::Serialize;
use time::{format_description::well_known::Rfc3339, OffsetDateTime};
use velo::{
    explorer::{NetParams, NetworkSampler},
    traffic_matrix::*,
    utils::{fit_parameters, FittedParameters},
};

const NUM_SEEDS: usize = 4;
const NUM_PREFIXES: u32 = 100_000;
const TARGET_DELTA: f64 = 0.051;
const STEP: f64 = 0.02;

type Writer = csv::Writer<std::fs::File>;
type Net = Network<SimplePrefix, BasicEventQueue<SimplePrefix>>;

#[derive(Debug, Clone, Serialize)]
struct Datapoint {
    /// Seed used to sample the traffic matrix
    tm_seed: u64,
    /// Number of destination prefixes
    num_prefixes: u32,
    /// The estimation for the repulsion factor based on the sampled traffic matrix, assuming it is
    /// LogNormal distributed
    repulsion_input: f64,
    /// The parameter for sampling the repulsion factor. This is the sigma parameter for a
    /// log-normal distribution.
    repulsion_output: f64,
    /// The estimation for the attraction factor based on the sampled traffic matrix, assuming it is
    /// LogNormal distributed
    attraction_input: f64,
    /// The parameter for sampling the friction factor. This is the sigma parameter of a log-normal
    /// distribution.
    attraction_output: f64,
    /// The estimation for the friction factor based on the sampled traffic matrix, assuming it is
    /// LogNormal distributed
    friction_input: f64,
    /// The parameter for sampling the friction factor. This is the sigma parameter of a log-normal
    /// distribution.
    friction_output: f64,
}

fn main() {
    let topo = TopologyZoo::TataNld;
    let num_externals = 30;
    let config_seed = 1;
    let net_params = NetParams {
        topo,
        num_externals,
        config_seed,
    };
    let net = net_params.sample().unwrap();

    let search_targets = vec![
        InputParameters::new(2.0, 3.0, 4.0),
        InputParameters::new(2.5, 3.0, 4.0),
        InputParameters::new(3.0, 3.0, 4.0),
        InputParameters::new(3.5, 3.0, 4.0),
        InputParameters::new(4.0, 3.0, 4.0),
        InputParameters::new(4.5, 3.0, 4.0),
        InputParameters::new(3.0, 2.0, 4.0),
        InputParameters::new(3.0, 2.5, 4.0),
        InputParameters::new(3.0, 3.5, 4.0),
        InputParameters::new(3.0, 4.0, 4.0),
        InputParameters::new(3.0, 4.5, 4.0),
        InputParameters::new(3.0, 3.0, 2.0),
        InputParameters::new(3.0, 3.0, 2.5),
        InputParameters::new(3.0, 3.0, 3.0),
        InputParameters::new(3.0, 3.0, 3.5),
        InputParameters::new(3.0, 3.0, 4.5),
        InputParameters::new(3.0, 3.0, 5.0),
    ];

    std::fs::create_dir_all("measurements").unwrap();
    let filename = format!(
        "measurements/tm-parameter-space-{}.csv",
        OffsetDateTime::now_utc().format(&Rfc3339).unwrap()
    );
    let writer = csv::WriterBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .from_path(&filename)
        .unwrap();
    let writer = Arc::new(Mutex::new(writer));
    let lut = Arc::new(Mutex::new(HashMap::new()));

    let num_main_threads = rayon::current_num_threads() / NUM_SEEDS;
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_main_threads)
        .build()
        .unwrap()
        .install(|| {
            search_targets.into_par_iter().for_each(|inp| {
                search_target(inp, &net, lut.clone(), writer.clone());
            })
        });
}

fn search_target(
    mut inp: InputParameters,
    net: &Net,
    lut: Arc<Mutex<HashMap<InputParameters, FittedParameters>>>,
    writer: Arc<Mutex<Writer>>,
) -> (InputParameters, FittedParameters) {
    let mut dim = Dimension::Attraction;
    let mut step = 0.0;
    let mut dist = 10.0;
    let target: FittedParameters = inp.into();
    let mut res: FittedParameters = inp.into();
    let mut visited = HashMap::<InputParameters, (f64, FittedParameters)>::new();
    while dist > TARGET_DELTA {
        // do the step
        inp.step(dim, step);
        // compute the next value
        res = get_parameter_distr(inp, net, lut.clone(), writer.clone());
        // get the update step
        (dist, dim, step) = get_dist(&res, &target);
        // insert in the visited array, and check for loops
        if visited.contains_key(&inp) {
            println!("WARN: could not converge!");
            // get the one with the smallest distance
            res = visited
                .values()
                .min_by_key(|(x, _)| NotNan::new(*x).unwrap())
                .map(|(_, res)| res.clone())
                .unwrap();
            break;
        }
        visited.insert(inp, (dist, res.clone()));
    }
    println!(
        "Found assignment:\n  Target {:?}\n  Input  {:?}\n  Output {:?}",
        target,
        FittedParameters::from(inp),
        res
    );

    (inp, res)
}

fn get_parameter_distr(
    inp: InputParameters,
    net: &Net,
    lut: Arc<Mutex<HashMap<InputParameters, FittedParameters>>>,
    writer: Arc<Mutex<Writer>>,
) -> FittedParameters {
    // check if we have already computed that thing
    {
        let lut = lut.lock().unwrap();
        if let Some(res) = lut.get(&inp) {
            return res.clone();
        }
    }

    // not yet computed! generate the parameters
    let res_vec =
        rayon::ThreadPoolBuilder::new()
            .num_threads(NUM_SEEDS)
            .build()
            .unwrap()
            .install(|| {
                let repeats =
                    (0..NUM_SEEDS).map(|x| (x as u64, writer.clone())).collect::<Vec<_>>();
                repeats
                    .into_par_iter()
                    .map(|(seed, w)| {
                        let out = get_parameters(inp, net, seed);
                        let datapoint = Datapoint {
                            tm_seed: seed,
                            num_prefixes: NUM_PREFIXES,
                            repulsion_input: inp.repulsion(),
                            repulsion_output: out.repulsion,
                            attraction_input: inp.attraction(),
                            attraction_output: out.attraction,
                            friction_input: inp.friction(),
                            friction_output: out.friction,
                        };
                        let mut wr = w.lock().unwrap();
                        wr.serialize(datapoint).unwrap();
                        wr.flush().unwrap();
                        out
                    })
                    .collect::<Vec<FittedParameters>>()
            });

    // compute the mean for res
    let n = res_vec.len() as f64;
    let res = FittedParameters {
        attraction: res_vec.iter().map(|x| x.attraction).sum::<f64>() / n,
        repulsion: res_vec.iter().map(|x| x.repulsion).sum::<f64>() / n,
        friction: res_vec.iter().map(|x| x.friction).sum::<f64>() / n,
    };
    {
        let mut lut = lut.lock().unwrap();
        lut.insert(inp, res.clone());
    }

    res
}

/// Compute a single sample
fn get_parameters<P: Prefix, Q>(
    p: InputParameters,
    net: &Network<P, Q>,
    seed: u64,
) -> FittedParameters {
    let mut tm_sampler = GravityTMSampler::new(net);
    tm_sampler
        .prefixes(NUM_PREFIXES)
        .seed(seed)
        .invert_friction(true)
        .renormalize_attraction(false)
        .attraction(log_normal(p.attraction()))
        .repulsion(log_normal(p.repulsion()))
        .friction(log_normal(p.friction()));
    let tm = tm_sampler.sample(false);
    fit_parameters(&tm)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
struct InputParameters {
    attraction_times_100: usize,
    repulsion_times_100: usize,
    friction_times_100: usize,
}

impl InputParameters {
    fn new(repulsion: f64, attraction: f64, friction: f64) -> Self {
        Self {
            attraction_times_100: (attraction * 100.0).round() as usize,
            repulsion_times_100: (repulsion * 100.0).round() as usize,
            friction_times_100: (friction * 100.0).round() as usize,
        }
    }

    fn attraction(&self) -> f64 {
        self.attraction_times_100 as f64 / 100.0
    }

    fn repulsion(&self) -> f64 {
        self.repulsion_times_100 as f64 / 100.0
    }

    fn friction(&self) -> f64 {
        self.friction_times_100 as f64 / 100.0
    }

    fn step(&mut self, dim: Dimension, step: f64) {
        match dim {
            Dimension::Attraction => {
                self.attraction_times_100 = ((self.attraction() + step) * 100.0).round() as usize
            }
            Dimension::Repulsion => {
                self.repulsion_times_100 = ((self.repulsion() + step) * 100.0).round() as usize
            }
            Dimension::Friction => {
                self.friction_times_100 = ((self.friction() + step) * 100.0).round() as usize
            }
        }
    }
}

impl From<InputParameters> for FittedParameters {
    fn from(value: InputParameters) -> Self {
        Self {
            attraction: value.attraction(),
            repulsion: value.repulsion(),
            friction: value.friction(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Dimension {
    Attraction,
    Repulsion,
    Friction,
}

/// Get the distance from a to b (infinity-norm). Also, get the dimension which has the largest
/// difference. Finally, get the next step in that direction to make a more like b
fn get_dist(current: &FittedParameters, target: &FittedParameters) -> (f64, Dimension, f64) {
    let da = target.attraction - current.attraction;
    let dr = target.repulsion - current.repulsion;
    let df = target.friction - current.friction;
    [
        (
            da.abs(),
            Dimension::Attraction,
            if da > 0.0 { STEP } else { -STEP },
        ),
        (
            dr.abs(),
            Dimension::Repulsion,
            if dr > 0.0 { STEP } else { -STEP },
        ),
        (
            df.abs(),
            Dimension::Friction,
            if df > 0.0 { STEP } else { -STEP },
        ),
    ]
    .into_iter()
    .max_by_key(|(x, _, _)| NotNan::new(*x).unwrap())
    .unwrap()
}
