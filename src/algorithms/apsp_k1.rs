//! Implementation of the all-pairs shortest path (APSP) algorthm to query shortest-path length
//! under single link failures in constant time. The algorithm follows the publication from
//! [Demetrescu et. al., SIAM'08](https://doi.org/10.1137/S0097539705429847).

use crate::algorithms::unique_shortest_paths;

use super::{dijkstra, shortest_path_tree::SourceSpt, EdgeList, GraphList, NodeList, UniqueWeight};

use petgraph::{
    prelude::*,
    stable_graph::IndexType,
    visit::{VisitMap, Visitable},
};

/// Oracle for requesting the shortest path distance for all pairs under single link failures
#[derive(Debug)]
pub struct ApspOracleK1<Ix> {
    x: NodeList<NodeList<OracleEntry<Ix>, Ix>, Ix>,
    edges: EdgeList<(NodeIndex<Ix>, NodeIndex<Ix>, UniqueWeight), Ix>,
}

#[derive(Debug)]
struct OracleEntry<Ix> {
    h: usize,
    d: UniqueWeight,
    dl: Vec<UniqueWeight>,
    dr: Vec<UniqueWeight>,
    sl: Vec<UniqueWeight>,
    sr: Vec<UniqueWeight>,
    vl: Vec<NodeIndex<Ix>>,
    vr: Vec<NodeIndex<Ix>>,
    de: UniqueWeight,
}

impl<Ix> Default for OracleEntry<Ix> {
    fn default() -> Self {
        Self {
            h: usize::MAX,
            d: UniqueWeight::MAX,
            dl: Default::default(),
            dr: Default::default(),
            sl: Default::default(),
            sr: Default::default(),
            vl: Default::default(),
            vr: Default::default(),
            de: UniqueWeight::MAX,
        }
    }
}

impl<Ix: IndexType> ApspOracleK1<Ix> {
    /// Create a new oracle.
    pub fn new<N: Copy>(g: &Graph<N, u64, Directed, Ix>) -> Self {
        let mut g = unique_shortest_paths(g);
        let edges = EdgeList::from_fn(&g, |e| {
            let (src, dst) = g.edge_endpoints(e).unwrap();
            let w = *g.edge_weight(e).unwrap();
            (src, dst, w)
        });

        let mut data: NodeList<NodeList<OracleEntry<Ix>, Ix>, Ix> = NodeList::new(&g);

        // compute all shortest paths
        let spt = NodeList::from_iter(&g, |root| SourceSpt::new(&g, root));

        // assign d and h based on the computed value of spt
        data.zip_mut(&spt, |data, spt| {
            data.zip_mut(spt.nodes(), |data, n| {
                data.h = n.dist;
                data.d = n.cost;
            })
        });

        // extract the apsp
        let apsp = NodeList::from_other(&data, |_, data| {
            NodeList::from_other(data, |_, data| data.d)
        });

        // compute dl, sl, and vl
        for (x, data) in data.idx_iter_mut() {
            let spt = &spt[x];

            // compute vl
            for (y, data) in data.idx_iter_mut() {
                let hxy = spt[y].dist;
                // push zero distance which is always the source
                data.vl.push(x);

                // go through all others. TODO result could be cached
                if hxy > 0 && hxy < usize::MAX {
                    for i in 1..=(1 + usize::ilog2(hxy)) {
                        let dist = 1usize << (i - 1);
                        data.vl.push(spt.node_at_dist(y, dist).unwrap())
                    }
                }
            }

            // compute dl
            for (y, data) in data.idx_iter_mut() {
                let hxy = spt[y].dist;
                // if hxy is infinity, then do nothing
                if hxy == usize::MAX {
                    continue;
                }
                // push dl[0] to be infinity
                data.dl.push(UniqueWeight::MAX);
                // iterate over all other elements that subpaths we need to avoid
                if hxy > 0 && hxy < usize::MAX {
                    for i in 1..=(usize::ilog2(hxy)) {
                        let lower = 1usize << (i - 1);
                        let upper = (1usize << i) - 1;
                        let subpath = spt.subpath(y, lower, upper);
                        data.dl.push(exclude(&g, x, y, &subpath));
                    }
                }
            }

            // compute sl
            for (y, data) in data.idx_iter_mut() {
                let hxy = spt[y].dist;
                // if hxy is infinity, then do nothing
                if hxy == usize::MAX {
                    continue;
                }
                // push sl[0] to be infinity
                data.sl.push(UniqueWeight::MAX);
                if hxy > 0 && hxy < usize::MAX {
                    for i in 1..(1 + ceil_ilog2(hxy)) {
                        let dist = 1usize << (i - 1);
                        let failed_node = spt.node_at_dist(y, dist).unwrap();
                        data.sl
                            .push(exclude_fast(&g, &apsp, spt, x, y, failed_node));
                    }
                }
            }
        }

        // reverse the graph
        g.reverse();

        // compute all shortest paths
        let spt = NodeList::from_iter(&g, |root| SourceSpt::new(&g, root));

        // compute dr, sr, and vr
        for (x, data) in data.idx_iter_mut() {
            // compute vl
            for (y, data) in data.idx_iter_mut() {
                let spt = &spt[y];
                let hxy = spt[x].dist;

                // push zero distance which is always the source
                data.vl.push(y);
                // go through all others. TODO result could be cached
                if hxy > 0 && hxy < usize::MAX {
                    for i in 1..=(1 + usize::ilog2(hxy)) {
                        let dist = 1usize << (i - 1);
                        data.vl.push(spt.node_at_dist(x, dist).unwrap())
                    }
                }
            }

            // compute dl
            for (y, data) in data.idx_iter_mut() {
                let spt = &spt[y];
                let hxy = spt[x].dist;

                // if hxy is infinity, then do nothing
                if hxy == usize::MAX {
                    continue;
                }
                // push dl[0] to be infinity
                data.dl.push(UniqueWeight::MAX);
                // iterate over all other elements that subpaths we need to avoid
                if hxy > 0 && hxy < usize::MAX {
                    for i in 1..=(usize::ilog2(hxy)) {
                        let lower = 1usize << (i - 1);
                        let upper = (1usize << i) - 1;
                        let subpath = spt.subpath(x, lower, upper);
                        data.dl.push(exclude(&g, y, x, &subpath));
                    }
                }
            }

            // compute sl
            for (y, data) in data.idx_iter_mut() {
                let spt = &spt[y];
                let hxy = spt[x].dist;

                // if hxy is infinity, then do nothing
                if hxy == usize::MAX {
                    continue;
                }
                // push sl[0] to be infinity
                data.sl.push(UniqueWeight::MAX);
                if hxy > 0 && hxy < usize::MAX {
                    for i in 1..(1 + ceil_ilog2(hxy)) {
                        let dist = 1usize << (i - 1);
                        let failed_node = spt.node_at_dist(x, dist).unwrap();
                        data.sl
                            .push(exclude_fast(&g, &apsp, spt, y, x, failed_node));
                    }
                }
            }
        }

        // compute de
        for (x, data) in data.idx_iter_mut() {
            for (y, data) in data.idx_iter_mut() {
                // compute the distance from x to y without the first edge on x.
                // First, get the first neighbor:
                if let Some(nh) = spt[y][x].prev {
                    // now, find the shortest path in G without that edge.
                    let e = g.find_edge(x, nh).unwrap();
                    let w = *g.edge_weight(e).unwrap();
                    g.remove_edge(e);

                    // compute dijkstra
                    let dist = dijkstra::dijkstra(&g, x, Some(y))[y];
                    data.de = dist;

                    // add back the edge
                    g.add_edge(x, nh, w);
                }
            }
        }

        Self { x: data, edges }
    }

    /// Query algorithm for failed nodes
    fn v_dist(&self, x: NodeIndex<Ix>, y: NodeIndex<Ix>, v: NodeIndex<Ix>) -> UniqueWeight {
        if x == v || y == v {
            return UniqueWeight::MAX;
        }

        let xv = &self.x[x][v];
        let xy = &self.x[x][y];
        let vy = &self.x[v][y];

        if xy.d == UniqueWeight::MAX {
            return UniqueWeight::MAX;
        }

        if xv.d + vy.d > xy.d {
            return xy.d;
        }

        let l = ceil_ilog2(xv.h);
        if xv.h.count_ones() == 1 {
            return xy.sl[l + 1];
        }
        let r = ceil_ilog2(vy.h);
        if vy.h.count_ones() == 1 {
            return xy.sr[r + 1];
        }
        let mid_u = xv.vr[l];
        let mid_v = vy.vl[r];
        let mut d = UniqueWeight::min(
            self.x[x][mid_u].d + self.x[mid_u][y].sl[l],
            self.x[x][mid_v].sr[r] + self.x[mid_v][y].d,
        );
        if xv.h <= vy.h {
            d = UniqueWeight::min(d, xy.dl[l])
        } else {
            d = UniqueWeight::min(d, xy.dr[r])
        };

        d
    }

    /// Query algorithm for failed edges
    fn e_dist(
        &self,
        x: NodeIndex<Ix>,
        y: NodeIndex<Ix>,
        u: NodeIndex<Ix>,
        v: NodeIndex<Ix>,
        w: UniqueWeight,
    ) -> UniqueWeight {
        if self.x[x][u].d + w + self.x[v][y].d > self.x[x][y].d {
            return self.x[x][y].d;
        }
        let d1 = self.v_dist(x, y, u);
        let d2 = self.x[x][u].d + self.x[u][y].de;
        return UniqueWeight::min(d1, d2);
    }

    /// Query the oracle.
    pub fn get_dist(
        &self,
        source: NodeIndex<Ix>,
        target: NodeIndex<Ix>,
        failed_edge: Option<EdgeIndex<Ix>>,
    ) -> Option<u64> {
        let w = if let Some(f) = failed_edge {
            let (u, v, w) = self.edges[f];
            self.e_dist(source, target, u, v, w)
        } else {
            self.x[source][target].d
        };
        if w < UniqueWeight::MAX {
            Some(w.into())
        } else {
            None
        }
    }
}

/// Compute the shortest path from `start` to all other nodes with
fn exclude<N: Copy, Ix: IndexType>(
    g: &Graph<N, UniqueWeight, Directed, Ix>,
    source: NodeIndex<Ix>,
    target: NodeIndex<Ix>,
    failed_path: &[NodeIndex<Ix>],
) -> UniqueWeight {
    let mut ignored = g.visit_map();
    failed_path.iter().for_each(|n| {
        ignored.visit(*n);
    });
    if ignored.is_visited(&target) || ignored.is_visited(&source) {
        return UniqueWeight::MAX;
    }
    let scores = dijkstra::node_failure(g, source, Some(target), ignored);
    scores[target]
}

fn exclude_fast<N, Ix: IndexType>(
    g: &Graph<N, UniqueWeight, Directed, Ix>,
    apsp: &NodeList<NodeList<UniqueWeight, Ix>, Ix>,
    spt: &SourceSpt<UniqueWeight, Ix>,
    x: NodeIndex<Ix>,
    y: NodeIndex<Ix>,
    root: NodeIndex<Ix>,
) -> UniqueWeight {
    if root == x || root == y {
        return UniqueWeight::MAX;
    }

    // build the subgraph according to algorithm 3.1 in the paper.
    let subtree = spt.subtree(root);
    let nodes_w = NodeList::from_other(subtree.nodes(), |_, d| d.dist > 0 && d.dist < usize::MAX);

    // prepare the sub_g from all edges in nodes_w
    let mut sub_g = g.filter_map(
        |_, _| Some(()),
        |e, w| {
            let (a, b) = g.edge_endpoints(e).unwrap();
            (nodes_w[a] && nodes_w[b]).then_some(*w)
        },
    );

    // add edges from x to all nodes in w
    for (b, _) in nodes_w.idx_iter().filter(|(_, b)| **b) {
        sub_g.add_edge(
            x,
            b,
            g.edges_directed(b, Direction::Incoming)
                .filter(|e| !nodes_w[e.source()])
                .map(|e| apsp[x][e.source()] + *e.weight())
                .min()
                .unwrap_or(UniqueWeight::MAX),
        );
    }

    // find the shortest path in the resulting graph sub_g
    dijkstra::dijkstra(&sub_g, x, Some(y))[y]
}

fn ceil_ilog2(x: usize) -> usize {
    if x.count_ones() == 1 {
        x.ilog2() as usize
    } else {
        x.ilog2() as usize + 1usize
    }
}

#[cfg(test)]
mod test {
    use proptest::proptest;
    use std::collections::HashMap;

    use crate::algorithms::apsp;

    use super::*;

    #[test]
    fn construct() {
        let g = graph_from_string("0:1:1");
        let oracle = ApspOracleK1::new(&g);
        let v0 = NodeIndex::<u16>::from(0);
        let v1 = NodeIndex::<u16>::from(1);
        println!("{:#?}", oracle.x[v0][v1]);
        assert_eq!(oracle.x[v0][v1].h, 1);
        assert_eq!(u64::from(oracle.x[v0][v1].d), 1);
        assert_eq!(oracle.v_dist(v0, v1, v1), UniqueWeight::MAX);
        assert_eq!(oracle.get_dist(v0, v1, Some(EdgeIndex::from(0))), None);
    }

    fn graph_from_string(s: &str) -> Graph<(), u64, Directed, u16> {
        let mut g = Graph::<(), u64, Directed, u16>::default();
        let mut lut = HashMap::new();
        for edge in s.split(";") {
            let edge = edge.trim();
            if edge.is_empty() {
                continue;
            }
            let (src, dst_cost) = edge.split_once(":").unwrap();
            let (dst, cost) = dst_cost.split_once(":").unwrap();
            let cost: u64 = cost.parse().unwrap();
            let src_id = *lut.entry(src).or_insert_with(|| g.add_node(()));
            let dst_id = *lut.entry(dst).or_insert_with(|| g.add_node(()));
            if src_id != dst_id && g.find_edge(src_id, dst_id).is_none() {
                g.add_edge(src_id, dst_id, cost);
            }
        }
        g
    }

    fn check_no_failure(s: &str) {
        let graph = graph_from_string(s);
        let oracle = ApspOracleK1::new(&graph);

        // check that the non-failure case is correct
        for (src, x) in apsp(&graph).idx_iter() {
            for (dst, w) in x.idx_iter() {
                let w = if *w == u64::MAX { None } else { Some(*w) };
                assert_eq!(w, oracle.get_dist(src, dst, None))
            }
        }
    }

    fn check_k1_failure(s: &str) {
        let mut graph = graph_from_string(&s);
        let oracle = ApspOracleK1::new(&graph);

        // for each edge failure
        for e in graph.edge_indices() {
            // set the edge weight
            let (u, v) = graph.edge_endpoints(e).unwrap();
            let old_w = *graph.edge_weight(e).unwrap();
            graph.remove_edge(e);

            // check that the non-failure case is correct
            for (src, x) in apsp(&graph).idx_iter() {
                for (dst, w) in x.idx_iter() {
                    let w = if *w == u64::MAX { None } else { Some(*w) };
                    assert_eq!(w, oracle.get_dist(src, dst, None))
                }
            }

            graph.add_edge(u, v, old_w);
        }
    }

    proptest! {
        #[test]
        fn no_failure(s in "([0-9]:[0-9]:[1-9][0-9]{0,2};){1,20}") {
            check_no_failure(&s)
        }

        #[test]
        fn k1_failure(s in "([0-9]:[0-9]:[1-9][0-9]{0,2};){1,20}") {
            check_k1_failure(&s)
        }
    }
}
