//! Shortest-Path DAG datastructures that store the DAG and the topological order.

use std::{borrow::Borrow, cmp::Ordering};

use itertools::Itertools;
use ordered_float::NotNan;
use petgraph::{algo::BoundedMeasure, graph::IndexType, prelude::*, EdgeType};

use super::{dijkstra, EdgeList, GraphList, NodeList};

/// A single-destination shortest-path DAG. The root is the destination of all paths.
#[derive(Debug, PartialEq, Clone)]
pub struct ShortestPathDag<E, Ix> {
    d: NodeList<DagNode<E, Ix>, Ix>,
    roots: Vec<NodeIndex<Ix>>,
    toposort: Vec<NodeIndex<Ix>>,
}

/// Data kept for each node in a Shortest Path DAG.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DagNode<E, Ix> {
    /// Next-hops from each node on their shortest paths towards the `root`.
    pub next: Vec<(NodeIndex<Ix>, EdgeIndex<Ix>)>,
    /// Cost of each node to reach the root
    pub cost: E,
    /// Number of hops for each node to reach the root.
    pub dist: usize,
    /// The specific root node
    pub root: Option<NodeIndex<Ix>>,
}

impl<E: BoundedMeasure, Ix> Default for DagNode<E, Ix> {
    fn default() -> Self {
        Self {
            next: Vec::new(),
            cost: E::max(),
            dist: usize::MAX,
            root: None,
        }
    }
}

impl<E: BoundedMeasure + Copy, Ix: IndexType> ShortestPathDag<E, Ix> {
    /// Create a single-source shortest-path tree.
    pub fn new<N, D: EdgeType>(graph: &Graph<N, E, D, Ix>, root: NodeIndex<Ix>) -> Self {
        let failures = EdgeList::new(graph);
        dijkstra::shortest_path_dag(graph, root, &failures)
    }

    /// Create a single-source shortest-path tree under failures.
    pub fn new_with_failures<N, D: EdgeType>(
        graph: &Graph<N, E, D, Ix>,
        root: NodeIndex<Ix>,
        failures: &EdgeList<bool, Ix>,
    ) -> Self {
        dijkstra::shortest_path_dag(graph, root, failures)
    }

    /// Transform all costs of the shortest-path DAG. The function `f` must be homomorphic, i.e.,
    /// the shortest-paths with edge `E` must be the same as `E2`.
    pub fn map<E2>(self, f: impl Fn(E) -> E2) -> ShortestPathDag<E2, Ix> {
        ShortestPathDag {
            d: self.d.map(|_, x| DagNode {
                next: x.next,
                cost: f(x.cost),
                dist: x.dist,
                root: x.root,
            }),
            roots: self.roots,
            toposort: self.toposort,
        }
    }
}

impl<E: Copy + Ord, Ix: IndexType> ShortestPathDag<E, Ix> {
    /// Merge multiple DAGS into one. We require that the iterator contains at least one DAG.
    pub fn merge<'a, I>(dags: I) -> ShortestPathDag<E, Ix>
    where
        I: IntoIterator<Item = &'a ShortestPathDag<E, Ix>>,
        E: 'a,
    {
        let mut iter = dags.into_iter();
        let first = iter.next().unwrap();

        let mut roots = first.roots.clone();
        let mut d = first.d.clone();

        for dag in iter {
            roots.extend_from_slice(&dag.roots);
            d.zip_mut(&dag.d, |cur, new| match cur.cost.cmp(&new.cost) {
                // the current node reaches the roots faster than in the new dag. Ignore that result.
                Ordering::Less => {}
                // We can reach the target within the same distance. Extend `cur`
                Ordering::Equal => {
                    // extend the next-hop
                    cur.next =
                        cur.next.clone().into_iter().chain(new.next.clone()).unique().collect();
                    // update the minimum distance to the root
                    cur.dist = cur.dist.max(new.dist);
                    // update the root
                    cur.root = None;
                }
                // the new node can reach the target faster. Set `cur` to `new`
                Ordering::Greater => {
                    *cur = new.clone();
                }
            })
        }

        let toposort = d.idx().sorted_by(|a, b| d[b].dist.cmp(&d[a].dist)).collect();

        ShortestPathDag { d, roots, toposort }
    }
}

impl<Ix: IndexType> ShortestPathDag<NotNan<f64>, Ix> {
    /// Merge multiple DAGS into one. We require that the iterator contains at least one DAG.
    pub fn merge_unique<'a, I>(dags: I, min_diff: f64) -> ShortestPathDag<NotNan<f64>, Ix>
    where
        I: IntoIterator<Item = &'a ShortestPathDag<NotNan<f64>, Ix>>,
    {
        let mut iter = dags.into_iter();
        let first = iter.next().unwrap();

        let mut roots = first.roots.clone();
        let mut d = first.d.clone();

        for dag in iter {
            let cost = |n: &DagNode<NotNan<f64>, Ix>| {
                n.cost
                    + if let Some(root) = n.root {
                        NotNan::new(
                            (root.index() as f64) / ((dag.d.len() * 10 + 1) as f64) * min_diff,
                        )
                        .unwrap()
                    } else {
                        NotNan::new(0.0).unwrap()
                    }
            };
            d.zip_mut(&dag.d, |cur, new| match cost(cur).cmp(&cost(new)) {
                // the current node reaches the roots faster than in the new dag. Ignore that result.
                Ordering::Less => {}
                // We can reach the target within the same distance. Extend `cur`
                Ordering::Equal => {
                    // in this case, we assume that we have an infinite weight...
                    assert!(cur.cost.into_inner() > 1e12);
                    assert!(new.cost.into_inner() > 1e12);
                }
                // the new node can reach the target faster. Set `cur` to `new`
                Ordering::Greater => {
                    *cur = new.clone();
                }
            });
            roots.extend_from_slice(&dag.roots);
        }

        let toposort = d.idx().sorted_by(|a, b| d[b].dist.cmp(&d[a].dist)).collect();

        ShortestPathDag { d, roots, toposort }
    }
}

impl<E, Ix: IndexType> ShortestPathDag<E, Ix> {
    pub(super) fn from_raw(
        d: NodeList<DagNode<E, Ix>, Ix>,
        roots: Vec<NodeIndex<Ix>>,
        toposort: Vec<NodeIndex<Ix>>,
    ) -> Self {
        Self { d, roots, toposort }
    }

    /// Get an iterator over all node indices
    pub fn idx(&self) -> impl Iterator<Item = NodeIndex<Ix>> {
        self.d.idx()
    }

    /// Get the roots of the shortest path tree
    pub fn roots(&self) -> &[NodeIndex<Ix>] {
        &self.roots
    }

    /// Get the next-hops of `node` along the shortest path of `node` towards `root`.
    pub fn next(&self, node: impl Borrow<NodeIndex<Ix>>) -> &[(NodeIndex<Ix>, EdgeIndex<Ix>)] {
        &self.d[node].next
    }

    /// Get the distance of the shortest-path from `node` to the `root`.
    pub fn dist(&self, node: impl Borrow<NodeIndex<Ix>>) -> usize {
        self.d[node].dist
    }

    /// Get the cost of the shortest-path from `node` to the `root`.
    pub fn cost(&self, node: impl Borrow<NodeIndex<Ix>>) -> &E {
        &self.d[node].cost
    }

    /// Get the egress that the specific node chooses.
    pub fn egress(&self, node: impl Borrow<NodeIndex<Ix>>) -> Option<NodeIndex<Ix>> {
        self.d[node].root
    }

    /// Get a reference to the datastructure.
    pub fn nodes(&self) -> &NodeList<DagNode<E, Ix>, Ix> {
        &self.d
    }

    /// Get all nodes in topological order, starting with those furthest away from the root.
    pub fn toposort(&self) -> &[NodeIndex<Ix>] {
        &self.toposort
    }

    /// Check if `node` is spanned by the tree.
    pub fn contains(&self, node: NodeIndex<Ix>) -> bool {
        self.d[node].dist < usize::MAX
    }
}

impl<I, E, Ix> std::ops::Index<I> for ShortestPathDag<E, Ix>
where
    I: Borrow<NodeIndex<Ix>>,
    Ix: IndexType,
{
    type Output = DagNode<E, Ix>;

    fn index(&self, idx: I) -> &Self::Output {
        &self.d[idx]
    }
}

#[cfg(test)]
mod test {
    use petgraph::visit::{VisitMap, Visitable};
    use proptest::proptest;
    use std::collections::{HashMap, HashSet};

    use super::*;
    use crate::algorithms::{apsp, shortest_path_dag};

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

    fn test_dag(s: &str) {
        let g = graph_from_string(s);

        let apsp = apsp(&g);

        // repeat for each destination
        for destination in g.node_indices() {
            // compute the shortest-path rooted at source
            let dag = ShortestPathDag::new(&g, destination);
            let oracle = shortest_path_dag(&g, &apsp, destination);

            for node in g.node_indices() {
                // check the cost is equal
                assert_eq!(dag[node].cost, apsp[node][destination]);
                // check all next-hops are equal
                assert_eq!(
                    dag[node].next.iter().copied().collect::<HashSet<_>>(),
                    oracle.edges(node).map(|e| (e.target(), *e.weight())).collect::<HashSet<_>>(),
                    "from {node:?} to {destination:?}",
                );
            }
        }
    }

    #[test]
    fn manual() {
        test_toposort("7:3:1;6:2:1;3:6:1;0:2:100;0:5:1;3:2:10;5:3:1;");
    }

    fn test_toposort(s: &str) {
        let g = graph_from_string(s);

        // repeat for each destination
        for destination in g.node_indices() {
            // compute the shortest-path rooted at source
            let dag = ShortestPathDag::new(&g, destination);
            let mut visited = g.visit_map();

            for node in dag.toposort() {
                visited.visit(*node);
                // next must not be visited
                for (next, _) in &dag[node].next {
                    assert!(
                        !visited.is_visited(next),
                        "node {next:?} is already visited before {node:?} was (destination: {destination:?})"
                    );
                }
            }
        }
    }

    proptest! {
        #[test]
        fn dag(s in "([0-9]:[0-9]:[1-9][0-9]{0,2};){1,20}") {
            test_dag(&s)
        }

        #[test]
        fn toposort(s in "([0-9]:[0-9]:[1-9][0-9]{0,2};){1,20}") {
            test_toposort(&s)
        }
    }
}
