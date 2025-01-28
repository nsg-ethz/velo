//! Program to compare velo vs qarc

#![deny(missing_docs, missing_debug_implementations)]
#![allow(clippy::type_complexity)]

use std::{collections::HashMap, time::Instant};

use rand::prelude::*;
use serde::Serialize;
use time::{format_description::well_known::Rfc3339, OffsetDateTime};
use velo::{analysis::Velo, performance::TrafficClass, traffic_matrix::*};

use bgpsim::{prelude::*, topology_zoo::TopologyZoo};
use itertools::Itertools;

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();

    let filename = format!(
        "measurements/qarc-comparison-{}.csv",
        OffsetDateTime::now_utc().format(&Rfc3339).unwrap()
    );
    let mut writer = csv::WriterBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .from_path(filename)?;

    #[derive(Serialize)]
    struct Record {
        topo: TopologyZoo,
        num_nodes: usize,
        num_edges: usize,
        link_failures: usize,
        time: f64,
        /// Number of edges that cannot accurately reflect the variability in the worst-case.
        num_edges_inaccurate_variability: usize,
    }

    for topo in TopologyZoo::topologies_increasing_edges()
        .iter()
        .copied()
        .filter(|t| t.num_edges() * 2 <= 210)
    {
        let (net, tm, state) = topology::<SimplePrefix>(topo, 1, true);
        for link_failures in [1, 2] {
            let start = Instant::now();

            let mut config = Velo::new(&net);
            config.directional_link_failures();
            let velo = config.prepare(&state, &tm);
            let result = velo.analyze(Some(0), link_failures, true);

            let time = start.elapsed().as_secs_f64();

            writer.serialize(Record {
                topo,
                num_nodes: net.internal_indices().count(),
                num_edges: net.ospf_network().internal_edges().count(),
                link_failures,
                time,
                num_edges_inaccurate_variability: result.check_variability(&velo, false),
            })?;
            writer.flush()?;
        }
    }

    Ok(())
}

fn all_pairs_tm<P: Prefix, Q: EventQueue<P>, Ospf: OspfImpl>(
    net: &mut Network<P, Q, Ospf>,
    routers: Vec<RouterId>,
    att: impl Distribution<f64>,
    rep: impl Distribution<f64>,
    fri: impl Distribution<f64>,
    seed: u64,
) -> (TrafficMatrix<P>, HashMap<P, Vec<RouterId>>) {
    let targets = routers
        .iter()
        .map(|r| {
            let ext =
                net.add_external_router(format!("{}-ext", r.fmt(net)), AsId(r.index() as u32));
            net.add_link(*r, ext).unwrap();
            ext
        })
        .collect_vec();

    let mut rng = SmallRng::seed_from_u64(seed);

    let pfx = |r: RouterId| P::from(r.index() as u32);

    let reps: HashMap<RouterId, f64> = routers.iter().map(|r| (*r, rep.sample(&mut rng))).collect();
    let atts: HashMap<RouterId, f64> = targets.iter().map(|e| (*e, att.sample(&mut rng))).collect();

    let traffic_matrix: TrafficMatrix<P> = routers
        .iter()
        .flat_map(|s| targets.iter().map(move |t| (*s, *t)))
        .map(|(s, t)| {
            (
                TrafficClass {
                    src: s,
                    dst: pfx(t),
                },
                reps[&s] * atts[&t] * fri.sample(&mut rng),
            )
        })
        .collect();

    let current_state: HashMap<P, Vec<RouterId>> =
        targets.iter().map(|t| (pfx(*t), vec![*t])).collect();

    (traffic_matrix, current_state)
}

fn topology<P: Prefix>(
    topo: TopologyZoo,
    seed: u64,
    constant_link_weights: bool,
) -> (
    Network<P, BasicEventQueue<P>, GlobalOspf>,
    TrafficMatrix<P>,
    HashMap<P, Vec<RouterId>>,
) {
    let mut net = topo.build(Default::default());
    let mut rng = SmallRng::seed_from_u64(seed);
    if !constant_link_weights {
        net.build_link_weights_seeded(
            &mut rng,
            bgpsim::builder::uniform_integer_link_weight_seeded,
            (10, 10),
        )
        .unwrap();
    }
    let routers = net.internal_indices().collect_vec();
    let (tm, state) = all_pairs_tm(
        &mut net,
        routers,
        uniform(1.0, 10.0),
        uniform(1.0, 10.0),
        uniform(1.0, 2.0),
        rng.gen(),
    );

    (net, tm, state)
}
