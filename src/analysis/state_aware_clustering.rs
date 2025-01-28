//! Module to perform clustering while remainin

use bgpsim::prelude::*;
use itertools::Itertools;

use std::collections::{BTreeMap, HashMap};

use crate::{
    algorithms::{GraphList, NodeId, Topology},
    traffic_matrix::{
        ClusterMode, Clustering, Clusters, KMeans, PerPrefixDemand, SingleCluster,
        TrafficMatrixData,
    },
    MyProgressIterator,
};

use super::{ClusterSettings, NList, PrefixData, PrefixSettings, TUncertainty, Velo};

/// Cluster similar prefixes within each prefix equivalence class.
pub(super) fn cluster<'a, P: Prefix + Send + Sync + 'static>(
    config: &'a Velo<P>,
    pecs: BTreeMap<PrefixSettings<'a>, Vec<(P, NList<f64>)>>,
) -> (Vec<PrefixData<'a, P>>, f64) {
    match config.clustering {
        ClusterSettings::FixedNum { num, mode } => {
            let (pecs, (non_cluster_data, _)) = split(config, pecs);
            let (mut cluster_data, cluster_err) = cluster_fixed_num(config, pecs, num, mode);
            cluster_data.extend(non_cluster_data);
            check_normalized(&mut cluster_data);
            (cluster_data, cluster_err)
        }
        ClusterSettings::TargetError { target, mode } => {
            let (pecs, (non_cluster_data, _)) = split(config, pecs);
            let (mut cluster_data, cluster_err) = cluster_target_error(config, pecs, target, mode);
            cluster_data.extend(non_cluster_data);
            check_normalized(&mut cluster_data);
            (cluster_data, cluster_err)
        }
        ClusterSettings::None => {
            let (mut cluster_data, cluster_err) = no_cluster(pecs);
            check_normalized(&mut cluster_data);
            (cluster_data, cluster_err)
        }
    }
}

/// Check that each PrefixData is normalized. If not, then perform the normalization.
fn check_normalized<P>(pecs: &mut [PrefixData<'_, P>]) {
    for pec in pecs {
        let demand_sum = pec.current_norm_demand.iter().sum::<f64>();
        if approx::relative_eq!(demand_sum, pec.total_demand, max_relative = 1e-7) {
            // data is not yet normalized!
            pec.current_norm_demand.iter_mut().for_each(|x| *x /= pec.total_demand);
        } else if approx::relative_eq!(demand_sum, 1.0, max_relative = 1e-7) {
            // data is already normalized
        } else {
            panic!("Data is neither normalized, nor is summing up to its total demand! \n  Demand: {:?}\n   Total: {}", pec.current_norm_demand, pec.total_demand);
        }
    }
}

/// Split the pecs into those that we must cluster, and those that we don't need to cluster.
fn split<'a, P: Prefix>(
    config: &'a Velo<P>,
    pecs: BTreeMap<PrefixSettings<'a>, Vec<(P, NList<f64>)>>,
) -> (
    BTreeMap<PrefixSettings<'a>, Vec<(P, NList<f64>)>>,
    (Vec<PrefixData<'a, P>>, f64),
) {
    let mut tot = 0.0;
    let mut non_cluster_data = Vec::new();
    let mut out_pecs = BTreeMap::new();
    for (settings, prefixes) in pecs {
        if matches!(settings.uncertainty, TUncertainty::Egress { .. }) {
            out_pecs.insert(settings, prefixes);
        } else {
            let mut elem = PrefixData {
                current_norm_demand: NList::new(&config.topo),
                current_state: settings.current_state,
                prefixes: Vec::new(),
                total_demand: 0.0,
                te_paths: settings.te_paths,
                uncertainty: settings.uncertainty,
            };
            for (p, x) in prefixes {
                let sum = x.iter().sum::<f64>();
                elem.total_demand += sum;
                elem.prefixes.push((p, sum));
                elem.current_norm_demand.zip_mut(&x, |a, b| *a += *b);
            }
            tot += elem.total_demand;
            non_cluster_data.push(elem);
        }
    }
    (out_pecs, (non_cluster_data, tot))
}

fn cluster_target_error<'a, P: Prefix + Send + Sync + 'static>(
    config: &'a Velo<P>,
    pecs: BTreeMap<PrefixSettings<'a>, Vec<(P, NList<f64>)>>,
    target_error: f64,
    mode: ClusterMode,
) -> (Vec<PrefixData<'a, P>>, f64) {
    let mut pecs = pecs
        .into_iter()
        .map(|(settings, prefixes)| {
            let total = prefixes.iter().flat_map(|(_, x)| x).sum::<f64>();
            (settings, prefixes, total)
        })
        .collect_vec();
    pecs.sort_by_key(|(_, prefixes, _)| prefixes.len());

    let mut clustering_error = 0.0;
    let mut result = Vec::new();

    let mut remaining_traffic = pecs.iter().map(|(_, _, x)| x).sum::<f64>();

    'outer: for (settings, prefixes, total) in
        pecs.into_iter().my_progress_config("State-aware clustering", false, config)
    {
        let tm_data = prepare_tm_data(config, prefixes, total);

        // first, do the naive clustering on a single cluster
        let mut clusters = SingleCluster { mode }.cluster(&tm_data, false);

        let remaining_error = target_error - clustering_error;
        let goal = remaining_error / remaining_traffic;
        remaining_traffic -= tm_data.total_demand;

        if clusters.error_pos(mode) < goal {
            // found a solution with the naive clustering technique!
            let (prefix_data, new_error) =
                from_clusters(&tm_data, clusters, settings, &config.topo);
            result.extend(prefix_data);
            clustering_error += new_error;
            continue 'outer;
        }

        // slowly increase the number of clusters until we meet the target error
        let mut k = 1usize;
        let inc = (tm_data.num_prefixes / 1_000).max(1);
        loop {
            k += inc;
            let mut clusters = KMeans {
                k,
                mode,
                parallel: config.cluster_parallel,
                ..Default::default()
            }
            .cluster(&tm_data, false);

            if clusters.error_pos(mode) < goal {
                let (prefix_data, new_error) =
                    from_clusters(&tm_data, clusters, settings, &config.topo);
                // found a solution with the naive clustering technique!
                result.extend(prefix_data);
                clustering_error += new_error;
                continue 'outer;
            }
        }
    }

    (result, clustering_error)
}

fn cluster_fixed_num<'a, P: Prefix + Send + Sync + 'static>(
    config: &'a Velo<P>,
    pecs: BTreeMap<PrefixSettings<'a>, Vec<(P, NList<f64>)>>,
    num_clusters: usize,
    mode: ClusterMode,
) -> (Vec<PrefixData<'a, P>>, f64) {
    let mut pecs = pecs
        .into_iter()
        .map(|(settings, prefixes)| {
            let total = prefixes.iter().flat_map(|(_, x)| x).sum::<f64>();
            (settings, prefixes, total)
        })
        .collect_vec();
    pecs.sort_by_key(|(_, prefixes, _)| prefixes.len());

    let assign_by = pecs.iter().map(|(_, prefixes, total)| (*total, prefixes.len())).collect();
    let assignment = assign_clusters(num_clusters, assign_by);

    let mut clustering_error = 0.0;
    let mut result = Vec::new();

    for ((settings, prefixes, total), num_clusters) in pecs
        .into_iter()
        .zip(assignment)
        .my_progress_config("State-aware clustering", false, config)
    {
        let tm_data = prepare_tm_data(config, prefixes, total);

        let clusters = if num_clusters <= 1 {
            SingleCluster { mode }.cluster(&tm_data, false)
        } else {
            KMeans {
                k: num_clusters,
                parallel: config.cluster_parallel,
                mode,
                ..Default::default()
            }
            .cluster(&tm_data, false)
        };

        // here, we need to perform clustering
        let (prefix_data, new_error) = from_clusters(&tm_data, clusters, settings, &config.topo);
        result.extend(prefix_data);
        clustering_error += new_error;
    }

    (result, clustering_error)
}

fn assign_clusters(num_clusters: usize, groups: Vec<(f64, usize)>) -> Vec<usize> {
    if groups.len() >= num_clusters {
        return vec![1; groups.len()];
    }
    let mut tot = groups.iter().map(|(x, _)| *x).sum::<f64>();
    let mut num_free = num_clusters - groups.len();
    let mut takes_all = vec![false; groups.len()];

    let mut fractions: Vec<(usize, usize, f64)>;

    'main_loop: loop {
        fractions = Vec::new();
        for (i, (x, size)) in groups.iter().enumerate() {
            if takes_all[i] {
                // skip those that we already know take all
                continue;
            }
            let f = num_free as f64 * (x / tot);
            if f.floor() as usize + 1 >= *size {
                // assign all to that size
                takes_all[i] = true;
                tot -= *x;
                num_free -= *size - 1;
                continue 'main_loop;
            }
            fractions.push((i, f.floor() as usize, f - f.floor()));
        }
        break 'main_loop;
    }

    // sort the fractions by their remaining fraction
    fractions.sort_by(|(_, _, a), (_, _, b)| b.partial_cmp(a).unwrap());
    // compute the number of clusters that are still unassigned
    fractions.iter().for_each(|(_, c, _)| num_free -= *c);
    // Add the remaining clusters until we run out of additional clusters. Only do that on those for
    // which the size is still less than the currently assigned groups
    for idx in (0..fractions.len()).cycle() {
        if num_free == 0 {
            break;
        }
        let (i, c, _) = &mut fractions[idx];
        if *c + 1 < groups[*i].1 {
            *c += 1;
            num_free -= 1;
        }
    }
    let lut = fractions.into_iter().map(|(i, c, _)| (i, c)).collect::<HashMap<_, _>>();

    let assignment: Vec<_> = groups
        .iter()
        .enumerate()
        .map(|(i, (_, size))| if takes_all[i] { *size } else { lut[&i] + 1 })
        .collect();

    assert_eq!(assignment.iter().sum::<usize>(), num_clusters);
    assignment
}

/// Generate an unique PrefixData for each Prefix.
fn no_cluster<'a, P: Prefix>(
    pecs: BTreeMap<PrefixSettings<'a>, Vec<(P, NList<f64>)>>,
) -> (Vec<PrefixData<'a, P>>, f64) {
    (
        pecs.into_iter()
            .flat_map(|(settings, prefixes)| {
                prefixes.into_iter().map(move |(p, current_demand)| {
                    let total_demand = current_demand.iter().sum::<f64>();
                    PrefixData {
                        current_norm_demand: current_demand,
                        current_state: settings.current_state,
                        prefixes: vec![(p, total_demand)],
                        total_demand,
                        te_paths: settings.te_paths,
                        uncertainty: settings.uncertainty,
                    }
                })
            })
            .collect(),
        0.0,
    )
}

fn prepare_tm_data<P: Prefix>(
    config: &Velo<P>,
    prefixes: Vec<(P, NList<f64>)>,
    total: f64,
) -> TrafficMatrixData<P, NodeId> {
    TrafficMatrixData {
        num_prefixes: prefixes.len(),
        num_sources: config.topo.node_count(),
        source_lut: config.topo.node_indices().map(|r| (r, r.index())).collect(),
        prefix_lut: prefixes.iter().enumerate().map(|(i, (p, _))| (*p, i)).collect(),
        total_demand: total,
        data: prefixes
            .into_iter()
            .map(|(prefix, demand)| {
                let mut d = PerPrefixDemand {
                    prefix,
                    demand: demand.into(),
                    norm_demand: Vec::new(),
                    scale: 0.0,
                };
                d.normalize();
                d
            })
            .collect(),
    }
}

fn from_clusters<'a, P: Prefix>(
    data: &TrafficMatrixData<P, NodeId>,
    mut clusters: Clusters<'_, P>,
    settings: PrefixSettings<'a>,
    topo: &Topology,
) -> (Vec<PrefixData<'a, P>>, f64) {
    let mode = KMeans::default().mode;

    let clustering_error = clusters.error_pos(mode) * data.total_demand;

    let mut result = Vec::new();

    // construct the data
    for mut cluster in clusters.clusters.into_iter() {
        let demand = cluster.effective_demand(mode);
        if demand.is_empty() {
            continue;
        }
        assert_eq!(demand.len(), topo.node_count());
        let current_demand = NList::from(demand);

        let total_demand = cluster.members().iter().map(|x| x.scale).sum();
        let per_prefix_demand = cluster.members().iter().map(|x| (x.prefix, x.scale));

        result.push(PrefixData {
            current_norm_demand: current_demand,
            prefixes: per_prefix_demand.into_iter().map(|(p, x)| (p, x / total_demand)).collect(),
            total_demand,
            current_state: settings.current_state,
            te_paths: settings.te_paths,
            uncertainty: settings.uncertainty,
        });
    }

    (result, clustering_error)
}
