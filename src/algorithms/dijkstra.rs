//! Re-implementation of the dijkstra algorithm that returns a NodeList.

use std::{cmp::Ordering, collections::BinaryHeap};

use fixedbitset::FixedBitSet;
use petgraph::{
    algo::BoundedMeasure,
    graph::IndexType,
    prelude::*,
    visit::{VisitMap, Visitable},
    EdgeType,
};

use super::{
    shortest_path_dag::DagNode,
    shortest_path_tree::{DestinationSpt, DestinationSptNode, SourceSpt, SourceSptNode},
    EdgeList, GraphList, NodeList, ShortestPathDag,
};

/// Compute the dijkstra algorithm
#[inline]
pub fn dijkstra<N, E: BoundedMeasure + Copy, D: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, D, Ix>,
    source: NodeIndex<Ix>,
    target: Option<NodeIndex<Ix>>,
) -> NodeList<E, Ix> {
    node_failure(graph, source, target, graph.visit_map())
}

/// Compute the shortest-path tree sourced at `root`.
pub fn shortest_path_tree<N, E: BoundedMeasure + Copy, D: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, D, Ix>,
    root: NodeIndex<Ix>,
) -> SourceSpt<E, Ix> {
    let mut d: NodeList<SourceSptNode<E, Ix>, Ix> = NodeList::new(graph);

    let mut visited = graph.visit_map();
    let mut visit_next = BinaryHeap::new();
    d[root].cost = E::min();
    d[root].dist = 0;
    visit_next.push(MinScored(E::min(), (root, 0)));

    while let Some(MinScored(node_score, (node, dist))) = visit_next.pop() {
        if visited.is_visited(&node) {
            continue;
        }
        for edge in graph.edges(node) {
            let next = edge.target();
            if visited.is_visited(&next) {
                continue;
            }
            let next_cost = node_score + *edge.weight();
            let old_score = d[next].cost;
            if old_score < E::max() {
                // occupied
                if next_cost < old_score {
                    d[next].cost = next_cost;
                    d[next].prev = Some(node);
                    d[next].dist = dist + 1;
                    visit_next.push(MinScored(next_cost, (next, dist + 1)));
                    //predecessor.insert(next.clone(), node.clone());
                }
            } else {
                // vacant
                d[next].cost = next_cost;
                d[next].prev = Some(node);
                d[next].dist = dist + 1;
                visit_next.push(MinScored(next_cost, (next, dist + 1)));
                //predecessor.insert(next.clone(), node.clone());
            }
        }
        visited.visit(node);
    }

    // compute next
    for n in d.idx() {
        if let Some(p) = d[n].prev {
            d[p].next.push(n)
        }
    }

    SourceSpt::from_raw(d, root)
}

/// Compute the shortest-path tree rooted at destination `root`.
pub fn shortest_path_dst_tree<N, E: BoundedMeasure + Copy, D: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, D, Ix>,
    root: NodeIndex<Ix>,
) -> DestinationSpt<E, Ix> {
    let mut d: NodeList<DestinationSptNode<E, Ix>, Ix> = NodeList::new(graph);

    let mut visited = graph.visit_map();
    let mut visit_next = BinaryHeap::new();
    d[root].cost = E::min();
    d[root].dist = 0;
    visit_next.push(MinScored(E::min(), (root, 0)));

    while let Some(MinScored(node_score, (node, dist))) = visit_next.pop() {
        if visited.is_visited(&node) {
            continue;
        }
        // walk the incoming edges
        for edge in graph.edges_directed(node, Direction::Incoming) {
            let next = edge.source();
            if visited.is_visited(&next) {
                continue;
            }
            let next_cost = node_score + *edge.weight();
            let old_score = d[next].cost;
            if old_score < E::max() {
                // occupied
                if next_cost < old_score {
                    d[next].cost = next_cost;
                    d[next].next = Some(node);
                    d[next].dist = dist + 1;
                    visit_next.push(MinScored(next_cost, (next, dist + 1)));
                    //predecessor.insert(next.clone(), node.clone());
                }
            } else {
                // vacant
                d[next].cost = next_cost;
                d[next].next = Some(node);
                d[next].dist = dist + 1;
                visit_next.push(MinScored(next_cost, (next, dist + 1)));
                //predecessor.insert(next.clone(), node.clone());
            }
        }
        visited.visit(node);
    }

    // compute prev
    for n in d.idx() {
        if let Some(p) = d[n].next {
            d[p].prev.push(n)
        }
    }

    DestinationSpt::from_raw(d, root)
}

/// Compute the shortest-path DAG towards destination `root`.
pub fn shortest_path_dag<N, E: BoundedMeasure + Copy, D: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, D, Ix>,
    root: NodeIndex<Ix>,
    failures: &EdgeList<bool, Ix>,
) -> ShortestPathDag<E, Ix> {
    let mut d: NodeList<DagNode<E, Ix>, Ix> = NodeList::new(graph);
    let mut toposort: Vec<NodeIndex<Ix>> = Vec::with_capacity(graph.node_count());

    // fill all nodes to point to the root
    d.iter_mut().for_each(|d| d.root = Some(root));

    let mut visited = graph.visit_map();
    let mut visit_next = BinaryHeap::new();
    d[root].cost = E::min();
    d[root].dist = 0;
    toposort.push(root);
    visit_next.push(MinScored(E::min(), root));

    while let Some(MinScored(node_score, node)) = visit_next.pop() {
        let dist = d[node].dist;
        if visited.is_visited(&node) {
            continue;
        }
        // walk the incoming edges
        for edge in graph.edges_directed(node, Direction::Incoming) {
            let e = edge.id();
            // ignore failed edges
            if failures[e] {
                continue;
            }
            let next = edge.source();
            if visited.is_visited(&next) {
                continue;
            }
            let next_cost = node_score + *edge.weight();
            let old_score = d[next].cost;
            if old_score < E::max() {
                // occupied
                if next_cost < old_score {
                    d[next].cost = next_cost;
                    d[next].next.clear();
                    d[next].next.push((node, e));
                    d[next].dist = dist + 1;
                    visit_next.push(MinScored(next_cost, next));
                } else if next_cost == old_score {
                    d[next].next.push((node, e));
                    d[next].dist = d[next].dist.max(dist + 1);
                }
            } else {
                // vacant
                d[next].cost = next_cost;
                d[next].next.push((node, e));
                d[next].dist = dist + 1;
                visit_next.push(MinScored(next_cost, next));
                // push the node in the toposort at the end (probably higher than all the others.)
                toposort.push(next);
            }
        }
        visited.visit(node);
    }

    toposort.reverse();
    toposort.sort_by(|a, b| d[b].dist.cmp(&d[a].dist));
    ShortestPathDag::from_raw(d, vec![root], toposort)
}

/// Compute the shortest-path three under a given node failure.
pub fn node_failure<N, E: BoundedMeasure + Copy, D: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, D, Ix>,
    source: NodeIndex<Ix>,
    target: Option<NodeIndex<Ix>>,
    ignored: FixedBitSet,
) -> NodeList<E, Ix> {
    let mut visited = ignored;
    let mut scores = NodeList::from_fn(graph, |_| E::max());
    //let mut predecessor = HashMap::new();
    let mut visit_next = BinaryHeap::new();
    let zero_score = E::default();
    scores[source] = zero_score;
    visit_next.push(MinScored(zero_score, source));
    while let Some(MinScored(node_score, node)) = visit_next.pop() {
        if visited.is_visited(&node) {
            continue;
        }
        for edge in graph.edges(node) {
            let next = edge.target();
            if visited.is_visited(&next) {
                continue;
            }
            if target == Some(next) {
                return scores;
            }
            let next_score = node_score + *edge.weight();
            let old_score = scores[next];
            if old_score < E::max() {
                // occupied
                if next_score < old_score {
                    scores[next] = next_score;
                    visit_next.push(MinScored(next_score, next));
                    //predecessor.insert(next.clone(), node.clone());
                }
            } else {
                // vacant
                scores[next] = next_score;
                visit_next.push(MinScored(next_score, next));
                //predecessor.insert(next.clone(), node.clone());
            }
        }
        visited.visit(node);
    }
    scores
}

/// `MinScored<K, T>` holds a score `K` and a scored object `T` in
/// a pair for use with a `BinaryHeap`.
///
/// `MinScored` compares in reverse order by the score, so that we can
/// use `BinaryHeap` as a min-heap to extract the score-value pair with the
/// least score.
///
/// **Note:** `MinScored` implements a total order (`Ord`), so that it is
/// possible to use float types as scores.
#[derive(Copy, Clone, Debug)]
pub struct MinScored<K, T>(pub K, pub T);

impl<K: PartialOrd, T> PartialEq for MinScored<K, T> {
    #[inline]
    fn eq(&self, other: &MinScored<K, T>) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<K: PartialOrd, T> Eq for MinScored<K, T> {}

impl<K: PartialOrd, T> PartialOrd for MinScored<K, T> {
    #[inline]
    fn partial_cmp(&self, other: &MinScored<K, T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: PartialOrd, T> Ord for MinScored<K, T> {
    #[inline]
    fn cmp(&self, other: &MinScored<K, T>) -> Ordering {
        let a = &self.0;
        let b = &other.0;
        if a == b {
            Ordering::Equal
        } else if a < b {
            Ordering::Greater
        } else if a > b {
            Ordering::Less
        } else if a.ne(a) && b.ne(b) {
            // these are the NaN cases
            Ordering::Equal
        } else if a.ne(a) {
            // Order NaN less, so that it is last in the MinScore order
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}
