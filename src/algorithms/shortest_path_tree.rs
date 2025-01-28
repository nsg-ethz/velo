//! Implementation of the shortest path tree

use std::borrow::Borrow;

use petgraph::{algo::BoundedMeasure, graph::IndexType, prelude::*, EdgeType};

use super::{dijkstra, NodeList};

/// A single-source shortest-path tree.
#[derive(Debug)]
pub struct SourceSpt<E, Ix> {
    d: NodeList<SourceSptNode<E, Ix>, Ix>,
    root: NodeIndex<Ix>,
}

/// Data kept for each node in a Shortest Path Tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceSptNode<E, Ix> {
    /// possible next-hops for the current node on any shortest path from `root` another node.
    pub next: Vec<NodeIndex<Ix>>,
    /// The last hop on the shortest path from `root` to the current node.
    pub prev: Option<NodeIndex<Ix>>,
    /// The cost for `root` to reach the current node
    pub cost: E,
    /// The number of hops for `root` to reach the current node.
    pub dist: usize,
}

impl<E: BoundedMeasure, Ix> Default for SourceSptNode<E, Ix> {
    fn default() -> Self {
        Self {
            next: Vec::new(),
            prev: None,
            cost: E::max(),
            dist: usize::MAX,
        }
    }
}

impl<E: BoundedMeasure + Copy, Ix: IndexType> SourceSpt<E, Ix> {
    /// Create a single-source shortest-path tree.
    pub fn new<N, D: EdgeType>(graph: &Graph<N, E, D, Ix>, root: NodeIndex<Ix>) -> Self {
        dijkstra::shortest_path_tree(graph, root)
    }

    /// Create a subtree of `self` rooted at `root`.
    pub fn subtree(&self, root: NodeIndex<Ix>) -> Self {
        let mut d: NodeList<SourceSptNode<E, Ix>, Ix> =
            NodeList::from_other(&self.d, |_, _| Default::default());
        let r_cost = self.d[root].cost;
        let r_dist = self.d[root].dist;
        let mut todo = vec![root];
        while let Some(n) = todo.pop() {
            d[n].next = self.d[n].next.clone();
            d[n].prev = self.d[n].prev;
            d[n].cost = self.d[n].cost - r_cost;
            d[n].dist = self.d[n].dist - r_dist;
            for x in &self.d[n].next {
                todo.push(*x);
            }
        }
        d[root].prev = None;
        Self { d, root }
    }
}

impl<E, Ix: IndexType> SourceSpt<E, Ix> {
    pub(super) fn from_raw(d: NodeList<SourceSptNode<E, Ix>, Ix>, root: NodeIndex<Ix>) -> Self {
        Self { d, root }
    }

    /// Get an iterator over all node indices
    pub fn idx(&self) -> impl Iterator<Item = NodeIndex<Ix>> {
        self.d.idx()
    }

    /// Get the root of the shortest path tree
    pub fn root(&self) -> NodeIndex<Ix> {
        self.root
    }

    /// Get the predecessor of a node, i.e., the node closer to the root.
    pub fn prev(&self, node: impl Borrow<NodeIndex<Ix>>) -> Option<NodeIndex<Ix>> {
        self.d[node].prev
    }

    /// Get the next nodes in the shortest-path tree, i.e., those that are further away from the
    /// root.
    pub fn next(&self, node: impl Borrow<NodeIndex<Ix>>) -> &[NodeIndex<Ix>] {
        &self.d[node].next
    }

    /// Get the distance of the shortest-path from the root to `node`
    pub fn dist(&self, node: impl Borrow<NodeIndex<Ix>>) -> usize {
        self.d[node].dist
    }

    /// Get the cost of the shortest-path from the root to `node`
    pub fn cost(&self, node: impl Borrow<NodeIndex<Ix>>) -> &E {
        &self.d[node].cost
    }

    /// Get a reference to the datastructure.
    pub fn nodes(&self) -> &NodeList<SourceSptNode<E, Ix>, Ix> {
        &self.d
    }

    /// Check if `node` is spanned by the tree.
    pub fn contains(&self, node: NodeIndex<Ix>) -> bool {
        self.d[node].dist < usize::MAX
    }

    /// Get the node at distance `dist` from the root on the shortest path from `node` to the root.
    pub fn node_at_dist(&self, mut node: NodeIndex<Ix>, dist: usize) -> Option<NodeIndex<Ix>> {
        // early exit if `node` is closer to the root than `dist`.
        if self.d[node].dist < dist {
            return None;
        }

        // walk towards the root
        while self.d[node].dist > dist {
            node = self.d[node].prev?;
        }

        Some(node)
    }

    /// Get the subpath from `root` to `target` from distance `i` from `root` to `j`.
    pub fn subpath(&self, target: NodeIndex<Ix>, i: usize, j: usize) -> Vec<NodeIndex<Ix>> {
        // limit j to be at most the distance from root to the target.
        let mut dist = self.d[target].dist;
        let j = j.min(dist);

        // check bounds
        if i > j || !self.contains(target) {
            return Vec::new();
        }

        let path_len = 1 + j - i;
        let mut path = vec![NodeIndex::new(0); path_len];
        let mut idx = path_len;

        // walk closer to `root` until we are at j
        let mut n = target;
        while dist > j {
            n = self.d[n].prev.unwrap();
            dist -= 1;
        }

        // now, n is at `dist` j. Walk until we are at dist i
        idx -= 1;
        path[idx] = n;
        while self.d[n].dist > i {
            n = self.d[n].prev.unwrap();
            dist -= 1;
            idx -= 1;
            path[idx] = n;
        }

        debug_assert_eq!(idx, 0);
        path
    }

    /// Get the shortest path from `root` to `target`
    pub fn path(&self, target: NodeIndex<Ix>) -> Option<Vec<NodeIndex<Ix>>> {
        let mut dist = self.d[target].dist;
        if dist == usize::MAX {
            return None;
        } else if target == self.root {
            return Some(vec![target]);
        }

        let mut path = vec![NodeIndex::new(0); dist + 1];
        let mut n = target;

        path[dist] = n;
        while n != self.root {
            n = self.d[n].prev.unwrap();
            dist -= 1;
            path[dist] = n;
        }

        debug_assert_eq!(dist, 0);
        Some(path)
    }
}

impl<I, E, Ix> std::ops::Index<I> for SourceSpt<E, Ix>
where
    I: Borrow<NodeIndex<Ix>>,
    Ix: IndexType,
{
    type Output = SourceSptNode<E, Ix>;

    fn index(&self, idx: I) -> &Self::Output {
        &self.d[idx]
    }
}

/// A single-destination shortest-path tree. The root is the destination of all paths.
#[derive(Debug)]
pub struct DestinationSpt<E, Ix> {
    d: NodeList<DestinationSptNode<E, Ix>, Ix>,
    root: NodeIndex<Ix>,
}

/// Data kept for each node in a Shortest Path Tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DestinationSptNode<E, Ix> {
    /// The next-hop on the shortest path from the current node to the `root`.
    pub next: Option<NodeIndex<Ix>>,
    /// The set of nodes that use the current node as next-hop along their shortest path towards
    /// `root`.
    pub prev: Vec<NodeIndex<Ix>>,
    /// The cost for the current node to reach the `root`.
    pub cost: E,
    /// The number of hops for the current node to reach the `root`.
    pub dist: usize,
}

impl<E: BoundedMeasure, Ix> Default for DestinationSptNode<E, Ix> {
    fn default() -> Self {
        Self {
            next: None,
            prev: Vec::new(),
            cost: E::max(),
            dist: usize::MAX,
        }
    }
}

impl<E: BoundedMeasure + Copy, Ix: IndexType> DestinationSpt<E, Ix> {
    /// Create a single-source shortest-path tree.
    pub fn new<N, D: EdgeType>(graph: &Graph<N, E, D, Ix>, root: NodeIndex<Ix>) -> Self {
        dijkstra::shortest_path_dst_tree(graph, root)
    }

    /// Create a subtree of `self` rooted at `root`.
    pub fn subtree(&self, root: NodeIndex<Ix>) -> Self {
        let mut d: NodeList<DestinationSptNode<E, Ix>, Ix> =
            NodeList::from_other(&self.d, |_, _| Default::default());
        let r_cost = self.d[root].cost;
        let r_dist = self.d[root].dist;
        let mut todo = vec![root];
        while let Some(n) = todo.pop() {
            d[n].next = self.d[n].next;
            d[n].prev = self.d[n].prev.clone();
            d[n].cost = self.d[n].cost - r_cost;
            d[n].dist = self.d[n].dist - r_dist;
            for x in &self.d[n].prev {
                todo.push(*x);
            }
        }
        d[root].next = None;
        Self { d, root }
    }
}

impl<E, Ix: IndexType> DestinationSpt<E, Ix> {
    pub(super) fn from_raw(
        d: NodeList<DestinationSptNode<E, Ix>, Ix>,
        root: NodeIndex<Ix>,
    ) -> Self {
        Self { d, root }
    }

    /// Get an iterator over all node indices
    pub fn idx(&self) -> impl Iterator<Item = NodeIndex<Ix>> {
        self.d.idx()
    }

    /// Get the root of the shortest path tree
    pub fn root(&self) -> NodeIndex<Ix> {
        self.root
    }

    /// Get the predecessors of a node, i.e., the set of nodes for which `node` is the first hop
    /// along the shortest path towards the `root`.
    pub fn prev(&self, node: impl Borrow<NodeIndex<Ix>>) -> &[NodeIndex<Ix>] {
        &self.d[node].prev
    }

    /// Get the next-hop of `node` along the shortest path of `node` towards `root`.
    pub fn next(&self, node: impl Borrow<NodeIndex<Ix>>) -> Option<NodeIndex<Ix>> {
        self.d[node].next
    }

    /// Get the distance of the shortest-path from `node` to the `root`.
    pub fn dist(&self, node: impl Borrow<NodeIndex<Ix>>) -> usize {
        self.d[node].dist
    }

    /// Get the cost of the shortest-path from `node` to the `root`.
    pub fn cost(&self, node: impl Borrow<NodeIndex<Ix>>) -> &E {
        &self.d[node].cost
    }

    /// Get a reference to the datastructure.
    pub fn nodes(&self) -> &NodeList<DestinationSptNode<E, Ix>, Ix> {
        &self.d
    }

    /// Check if `node` is spanned by the tree.
    pub fn contains(&self, node: NodeIndex<Ix>) -> bool {
        self.d[node].dist < usize::MAX
    }

    /// Get the node at distance `dist` to the root on the shortest path from `node` to the root.
    pub fn node_at_dist(&self, mut node: NodeIndex<Ix>, dist: usize) -> Option<NodeIndex<Ix>> {
        // early exit if `node` is closer to the root than `dist`.
        if self.d[node].dist < dist {
            return None;
        }

        // walk towards the root
        while self.d[node].dist > dist {
            node = self.d[node].next?;
        }

        Some(node)
    }

    /// Get the subpath from `source` to `root` from distance `i` from `root` to `j`.
    pub fn subpath(&self, source: NodeIndex<Ix>, i: usize, j: usize) -> Vec<NodeIndex<Ix>> {
        // limit j to be at most the distance from root to the target.
        let mut dist = self.d[source].dist;
        let j = j.min(dist);

        // check bounds
        if i > j || !self.contains(source) {
            return Vec::new();
        }

        let path_len = 1 + j - i;
        let mut path = Vec::with_capacity(path_len);

        // walk closer to `root` until we are at j
        let mut n = source;
        while dist > j {
            n = self.d[n].next.unwrap();
            dist -= 1;
        }

        // now, n is at `dist` j. Walk until we are at dist i
        path.push(n);
        while self.d[n].dist > i {
            n = self.d[n].next.unwrap();
            dist -= 1;
            path.push(n)
        }

        path
    }

    /// Get the shortest path from `target` to `root`
    pub fn path(&self, target: NodeIndex<Ix>) -> Option<Vec<NodeIndex<Ix>>> {
        let dist = self.d[target].dist;
        if dist == usize::MAX {
            return None;
        } else if target == self.root {
            return Some(vec![target]);
        }

        let mut path = Vec::with_capacity(dist + 1);
        let mut n = target;

        path.push(n);
        while n != self.root {
            n = self.d[n].next.unwrap();
            path.push(n);
        }

        Some(path)
    }
}

impl<I, E, Ix> std::ops::Index<I> for DestinationSpt<E, Ix>
where
    I: Borrow<NodeIndex<Ix>>,
    Ix: IndexType,
{
    type Output = DestinationSptNode<E, Ix>;

    fn index(&self, idx: I) -> &Self::Output {
        &self.d[idx]
    }
}

#[cfg(test)]
mod test {
    use proptest::proptest;
    use std::collections::HashMap;

    use super::*;
    use crate::algorithms::{unique_shortest_paths, UniqueWeight};

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

    proptest! {
        #[test]
        fn source_spt(s in "([0-9]:[0-9]:[1-9][0-9]{0,2};){1,20}") {
            let g = graph_from_string(&s);
            let g = unique_shortest_paths(&g);

            // repeat for each source
            for source in g.node_indices() {
                // compute the shortest-path rooted at source
                let spt = SourceSpt::new(&g, source);

                for target in g.node_indices() {
                    let path = spt.path(target);
                    assert_eq!(
                        path,
                        petgraph::algo::astar(
                            &g,
                            source,
                            |x| x == target,
                            |e| *e.weight(),
                            |_| UniqueWeight::MIN
                        )
                            .map(|(_, p)| p)
                    );
                    if let Some(path) = path {
                        assert_eq!(path.len() - 1, spt[target].dist);
                    } else {
                        assert_eq!(usize::MAX, spt[target].dist);
                    }
                }
            }
        }

        #[test]
        fn destination_spt(s in "([0-9]:[0-9]:[1-9][0-9]{0,2};){1,20}") {
            let g = graph_from_string(&s);
            let g = unique_shortest_paths(&g);

            // repeat for each destination
            for destination in g.node_indices() {
                // compute the shortest-path with destination `destination`
                let spt = DestinationSpt::new(&g, destination);

                for source in g.node_indices() {
                    let path = spt.path(source);
                    assert_eq!(
                        path,
                        petgraph::algo::astar(
                            &g,
                            source,
                            |x| x == destination,
                            |e| *e.weight(),
                            |_| UniqueWeight::MIN
                        )
                            .map(|(_, p)| p)
                    );
                    if let Some(path) = path {
                        assert_eq!(path.len() - 1, spt[source].dist);
                    } else {
                        assert_eq!(usize::MAX, spt[source].dist);
                    }
                }
            }
        }
    }
}
