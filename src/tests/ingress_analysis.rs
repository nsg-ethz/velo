use std::collections::HashMap;

use approx::assert_relative_eq;
use bgpsim::prelude::*;
use itertools::Itertools;

use crate::{
    analysis::{Uncertainty, Velo},
    performance::TrafficClass,
    traffic_matrix::TrafficMatrix,
};

type P = SimplePrefix;
type Q = BasicEventQueue<P>;
type Ospf = GlobalOspf;
type Net = Network<P, Q, Ospf>;

fn prepare() -> (Net, Vec<RouterId>, Vec<RouterId>) {
    let mut net = Net::default();
    let mut int = Vec::new();
    let mut ext = Vec::new();

    int.push(net.add_router("r0"));
    int.push(net.add_router("r1"));
    int.push(net.add_router("r2"));
    int.push(net.add_router("r3"));
    int.push(net.add_router("r4"));
    int.push(net.add_router("r5"));
    ext.push(net.add_external_router("e0", AsId(0)));
    ext.push(net.add_external_router("e1", AsId(1)));
    ext.push(net.add_external_router("e2", AsId(2)));
    ext.push(net.add_external_router("e3", AsId(3)));
    ext.push(net.add_external_router("e4", AsId(4)));
    ext.push(net.add_external_router("e5", AsId(5)));

    net.add_links_from([
        (int[0], int[1]),
        (int[1], int[2]),
        (int[1], int[3]),
        (int[2], int[4]),
        (int[3], int[4]),
        (int[2], int[3]),
        (int[4], int[5]),
    ])
    .unwrap();
    net.add_links_from([
        (int[0], ext[0]),
        (int[1], ext[1]),
        (int[2], ext[2]),
        (int[3], ext[3]),
        (int[4], ext[4]),
        (int[5], ext[5]),
    ])
    .unwrap();

    net.build_ibgp_full_mesh().unwrap();
    net.build_ebgp_sessions().unwrap();

    (net, int, ext)
}

fn tm_even_distr(
    int: &[RouterId],
    ext: &[RouterId],
    prefixes: &[P],
) -> (HashMap<P, Vec<RouterId>>, TrafficMatrix<P>) {
    let tm: TrafficMatrix<P> = prefixes
        .iter()
        .flat_map(|p| int.iter().copied().map(move |r| (r, p)))
        .map(|(src, &dst)| (TrafficClass { src, dst }, 1.0))
        .collect();

    let state: HashMap<P, Vec<RouterId>> =
        prefixes.iter().zip(ext).map(|(&p, &ext)| (p, vec![ext])).collect();

    (state, tm)
}

#[test]
fn no_changes() {
    let (net, int, ext) = prepare();
    let prefixes = ext.iter().map(|e| P::from(e.index() as u32)).collect::<Vec<_>>();
    let mut velo = Velo::new(&net);
    let (current_state, traffic_matrix) = tm_even_distr(&int, &ext, &prefixes);

    velo.hide_progress();
    for &p in &prefixes {
        velo.with_uncertainty(
            [p],
            Uncertainty::Ingress {
                ingresses: int.iter().copied().collect(),
            },
        );
    }

    let result = velo.prepare(&current_state, &traffic_matrix).analyze(Some(0), 0, false);

    assert_relative_eq!(result.loads[&(int[0], int[1])], 5.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[1], int[0])], 5.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[1], int[2])], 4.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[2], int[1])], 4.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[1], int[3])], 4.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[3], int[1])], 4.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[2], int[3])], 1.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[3], int[2])], 1.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[2], int[4])], 4.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[4], int[2])], 4.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[3], int[4])], 4.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[4], int[3])], 4.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[4], int[5])], 5.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[5], int[4])], 5.0, epsilon = 1e-7);
}

#[test]
fn single_change() {
    let (net, int, ext) = prepare();
    let prefixes = ext.iter().map(|e| P::from(e.index() as u32)).collect::<Vec<_>>();
    let mut velo = Velo::new(&net);
    let (current_state, traffic_matrix) = tm_even_distr(&int, &ext, &prefixes);

    velo.hide_progress();
    for &p in &prefixes {
        velo.with_uncertainty(
            [p],
            Uncertainty::Ingress {
                ingresses: int.iter().copied().collect(),
            },
        );
    }

    let velo_analysis = velo.prepare(&current_state, &traffic_matrix);
    let result = velo_analysis.analyze(Some(1), 0, true);

    println!("{:#?}", result.loads[&(int[0], int[1])]);
    println!("{:#?}", result.states[&(int[0], int[1])]);
    let modified_prefix = result.states[&(int[0], int[1])]
        .routing_inputs
        .as_ref()
        .map(|x| x.keys())
        .into_iter()
        .flatten()
        .next()
        .unwrap();
    println!(
        "{:#?}",
        velo_analysis.data.iter().find(|x| x
            .prefixes
            .iter()
            .map(|(p, _)| p)
            .contains(modified_prefix))
    );

    assert_relative_eq!(result.loads[&(int[0], int[1])], 10.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[1], int[0])], 6.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[1], int[2])], 8.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[2], int[1])], 8.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[1], int[3])], 8.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[3], int[1])], 8.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[2], int[3])], 6.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[3], int[2])], 6.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[2], int[4])], 8.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[4], int[2])], 8.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[3], int[4])], 8.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[4], int[3])], 8.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[4], int[5])], 6.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[5], int[4])], 10.0, epsilon = 1e-7);
}

#[test]
fn all_changes() {
    let (net, int, ext) = prepare();
    let prefixes = ext.iter().map(|e| P::from(e.index() as u32)).collect::<Vec<_>>();
    let mut velo = Velo::new(&net);
    let (current_state, traffic_matrix) = tm_even_distr(&int, &ext, &prefixes);

    velo.hide_progress();
    for &p in &prefixes {
        velo.with_uncertainty(
            [p],
            Uncertainty::Ingress {
                ingresses: int.iter().copied().collect(),
            },
        );
    }

    let result = velo.prepare(&current_state, &traffic_matrix).analyze(None, 0, false);

    assert_relative_eq!(result.loads[&(int[0], int[1])], 30.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[1], int[0])], 6.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[1], int[2])], 12.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[2], int[1])], 12.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[1], int[3])], 12.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[3], int[1])], 12.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[2], int[3])], 6.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[3], int[2])], 6.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[2], int[4])], 12.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[4], int[2])], 12.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[3], int[4])], 12.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[4], int[3])], 12.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[4], int[5])], 6.0, epsilon = 1e-7);
    assert_relative_eq!(result.loads[&(int[5], int[4])], 30.0, epsilon = 1e-7);
}
