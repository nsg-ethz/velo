//! Module to add traffic variability (or check whether it can be added without overapproximation)

use bgpsim::types::Prefix;

use crate::algorithms::{
    EdgeId, EdgeList, GraphList, NodeId, NodeList, ShortestPathDag, TopologyType,
};

use super::{PerformanceReport, VeloAnalysis};

impl<P: Prefix> PerformanceReport<P> {
    /// Add traffic variability to the performance report. The function returns a number of of
    /// edges; for how many edges is the variability an overapproximation.
    pub fn add_variability(
        &mut self,
        velo: &VeloAnalysis<'_, P>,
        additional_traffic: f64,
        invent_new: bool,
    ) -> usize {
        let num_inaccurate = self.check_variability(velo, invent_new);
        for x in self.loads.values_mut() {
            if *x > 0.0 {
                *x += additional_traffic
            }
        }
        num_inaccurate
    }

    /// Check if we can safely add traffic variability. An edge cannot accurately model the traffic
    /// variability, if:
    /// - The link has non-zero worst-case load, and
    /// - there does not exist any flow that sends all its traffic over that link.
    ///
    /// This function returns the number of such links.
    pub fn check_variability(&self, velo: &VeloAnalysis<'_, P>, invent_new: bool) -> usize {
        let topo = &velo.config.topo;
        let mut num_accurate = 0;

        for ((src, dst), worst_case_state) in &self.states {
            // skip if the link has worst-case load of 0
            if self.loads.get(&(*src, *dst)).copied().unwrap_or_default() == 0.0 {
                num_accurate += 1;
                continue;
            }

            // generate the failure state
            let mut failures = EdgeList::new(topo);
            for (a, b) in &worst_case_state.failures {
                let e = topo.topo_edge_id(*a, *b).unwrap();
                failures[e] = true;
            }

            let e = topo.topo_edge_id(*src, *dst).unwrap();

            // iterate over all source nodes
            'search: for &dst in &velo.config.egress_routers {
                let dst = topo.topo_id(dst);
                // compute the shortest path DAG towares that dst
                let dag = ShortestPathDag::new_with_failures(topo, dst, &failures);

                // remember which nodes we have visited
                let mut visited: NodeList<bool, _> = NodeList::new(topo);
                // go through each source router in topological order
                for &src in dag.toposort() {
                    // skip if we have already visited that node.
                    if visited[src] {
                        continue;
                    }
                    // skip if we cannot invent new sources, and the source does not generate any
                    // traffic
                    if !invent_new && velo.data.iter().all(|x| x.current_norm_demand[src] == 0.0) {
                        continue;
                    }

                    if unique_path_via(src, e, &dag, &mut visited) {
                        num_accurate += 1;
                        break 'search;
                    }
                }
            }
        }

        self.states.len() - num_accurate
    }
}

/// Function that checks if there exists an unique path from the source to the root of the dag via
/// the target edge. All nodes along its path are marked in `visited`. If there exists such a path,
/// then the function returns `true`.
fn unique_path_via<E>(
    src: NodeId,
    target_edge: EdgeId,
    dag: &ShortestPathDag<E, TopologyType>,
    visited: &mut NodeList<bool, TopologyType>,
) -> bool {
    let mut n = src;
    loop {
        visited[n] = true;
        let next = dag.next(n);
        if next.len() != 1 {
            return false;
        }
        let (next_node, next_edge) = next[0];
        if next_edge == target_edge {
            return true;
        }
        n = next_node;
    }
}
