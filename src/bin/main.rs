//! Verify Isp PERformance properties of networks.

#![deny(missing_docs, missing_debug_implementations)]

use std::collections::{HashMap, HashSet};

use velo::{
    analysis::{current_link_load, ClusterSettings, PerformanceReport, Velo},
    scenario::*,
    traffic_matrix::*,
};

use anyhow::Context;
use bgpsim::{
    network::Network,
    ospf::OspfImpl,
    prelude::NetworkFormatter,
    topology_zoo::TopologyZoo,
    types::{Prefix, RouterId},
};
use itertools::Itertools;
use std::fmt::Write;

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();

    // let (net, _graph) = ScenarioBuilder::new(ScenarioTopo::TopologyZoo(TopologyZoo::Kdl))
    // let (net, _graph) = ScenarioBuilder::new(ScenarioTopo::TopologyZoo(TopologyZoo::Abilene))
    // let (net, _graph) = ScenarioBuilder::new(ScenarioTopo::TopologyZoo(TopologyZoo::Colt))
    let net = ScenarioBuilder::new(ScenarioTopo::TopologyZoo(TopologyZoo::Deltacom))
        .seed(2)
        .external_routers(50)
        .build()
        .context("Generating the scenairo")?;

    // let mut tm_sampler = IidTMSampler::new(&net);
    // tm_sampler.prefixes(10_000).distr(uniform(0.0, 1.0));
    let mut tm_sampler = GravityTMSampler::new(&net);
    tm_sampler
        .prefixes(100_000)
        .seed(1)
        // .repulsion(log_normal(1.0))
        .repulsion(uniform(1.0, 8.0))
        // .attraction(log_normal(4.1))
        .attraction(pareto(0.9))
        .friction(uniform(1.0, 2.0));
    let traffic_matrix = tm_sampler.sample(true);
    let current_state = tm_sampler.current_state(None, true);
    let total_traffic = traffic_matrix.values().sum();

    // print TM statistics
    TrafficMatrixData::new(&traffic_matrix, true).print_statistics();

    let current_link_load = current_link_load(&net, &current_state, &traffic_matrix);

    let mut config = Velo::new(&net);
    // config.only_heaviest_prefixes(300);
    config.with_clustering(ClusterSettings::FixedNum {
        num: 300,
        mode: ClusterMode::NormalizedScaled,
    });
    let velo = config.prepare(&current_state, &traffic_matrix);

    let result_exact = velo.analyze(None, 0, false);
    let result_k1 = velo.analyze(Some(1), 0, false);
    let result_k10 = velo.analyze(Some(10), 0, false);

    print_result(
        &current_link_load,
        vec![result_k1, result_k10, result_exact],
        &net,
        total_traffic,
    );

    Ok(())
}

fn print_result<P: Prefix, Q, Ospf: OspfImpl>(
    current_link_load: &HashMap<(RouterId, RouterId), f64>,
    results: Vec<PerformanceReport<P>>,
    net: &Network<P, Q, Ospf>,
    total_traffic: f64,
) {
    // combine the results
    let mut edges = current_link_load.keys().copied().collect::<HashSet<_>>();
    for result in &results {
        edges.extend(result.loads.keys().copied());
    }

    let bounds = results
        .iter()
        .map(|result| {
            let pos_bound_rel = result.pos_bounds * 100.0 / total_traffic;
            let neg_bound_rel = result.neg_bounds * 100.0 / total_traffic;

            if approx::relative_eq!(pos_bound_rel, neg_bound_rel, epsilon = 1e-9) {
                format!("± {pos_bound_rel:.1}%")
            } else {
                let mut s = String::new();
                if pos_bound_rel > 1e-9 {
                    write!(&mut s, "+{pos_bound_rel:.1}%").unwrap();
                }
                if neg_bound_rel > 1e-9 {
                    if !s.is_empty() {
                        s.push(' ');
                    }
                    write!(&mut s, "-{neg_bound_rel:.1}%").unwrap();
                }
                s
            }
        })
        .collect_vec();

    for (src, dst) in edges {
        let key = (src, dst);

        let current = current_link_load.get(&key).copied().unwrap_or(0.0);
        let current_rel = current * 100.0 / total_traffic;

        print!(
            "{: >20} -> {: <20}: {}",
            src.fmt(net),
            dst.fmt(net),
            fmt_perc(current_rel),
        );

        for (result, bound) in results.iter().zip(bounds.iter()) {
            let worst_case = result.loads.get(&key).copied().unwrap_or(0.0);
            let worst_case_rel = worst_case * 100.0 / total_traffic;

            print!(" | {} {bound}", fmt_perc(worst_case_rel),);
        }

        println!()
    }
}

fn fmt_perc(p: f64) -> String {
    if approx::relative_eq!(p, 0.0, epsilon = 1e-9) {
        "  0%    ".to_string()
    } else {
        format!("{p: >7.3}%")
    }
}
