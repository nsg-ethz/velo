//! Module to translate a network into a graph and operate on it.

use std::{borrow::Borrow, collections::HashMap};

use bgpsim::prelude::*;
use bimap::BiHashMap;
use itertools::Itertools;
use petgraph::prelude::*;

use super::EdgeList;

/// The topology type for representing networks as graphs.
pub type TopologyType = u16;
/// The ID of a node within the graph.
pub type NodeId = NodeIndex<TopologyType>;
/// The ID of an edge within the graph.
pub type EdgeId = EdgeIndex<TopologyType>;

/// Physical topology, represented as a graph.
#[derive(Debug, Clone)]
pub struct Topology {
    /// The topology, stored as a directed graph.
    pub graph: Graph<(), u64, Directed, TopologyType>,
    lut: BiHashMap<RouterId, NodeId>,
}

impl std::ops::Deref for Topology {
    type Target = Graph<(), u64, Directed, TopologyType>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl Topology {
    /// Create a new topology from a network.
    pub fn new<P: Prefix, Q, Ospf: OspfImpl>(net: &Network<P, Q, Ospf>) -> Topology {
        let graph = net.get_topology();
        let mut graph = Graph::with_capacity(graph.node_count(), graph.edge_count());
        let mut lut: BiHashMap<RouterId, NodeId> = BiHashMap::with_capacity(graph.node_count());

        for n in net.internal_indices() {
            let n_prime = graph.add_node(());
            lut.insert(n, n_prime);
        }
        for e in net.ospf_network().internal_edges() {
            if let (Some(s_prime), Some(t_prime)) =
                (lut.get_by_left(&e.src), lut.get_by_left(&e.dst))
            {
                graph.add_edge(*s_prime, *t_prime, e.weight.round() as u64);
            }
        }

        Self { graph, lut }
    }

    /// Apply the function to all link weights. The argument of the function will be the current
    /// link weight, and the function should return a new link weight.
    pub fn map_link_weights<F>(&mut self, mut f: F)
    where
        F: FnMut(u64) -> u64,
    {
        self.graph.edge_weights_mut().for_each(|x| *x = f(*x))
    }

    /// Lookup the RouterId from a NodeId
    pub fn net_id(&self, id: impl Borrow<NodeId>) -> RouterId {
        *self.lut.get_by_right(id.borrow()).unwrap()
    }

    /// Lookup the node connecting a and b
    pub fn topo_edge_id(
        &self,
        a: impl Borrow<RouterId>,
        b: impl Borrow<RouterId>,
    ) -> Option<EdgeId> {
        let a = self.topo_id(a);
        let b = self.topo_id(b);
        self.graph.edges_connecting(a, b).next().map(|x| x.id())
    }

    /// Lookup the NodeId from a RouterId
    pub fn topo_id(&self, id: impl Borrow<RouterId>) -> NodeId {
        *self.lut.get_by_left(id.borrow()).unwrap()
    }

    /// Translate a path (list of router IDs) to a list of edges in the topo.
    pub fn path<I: IntoIterator<Item = R>, R: Borrow<RouterId>>(
        &self,
        path: I,
    ) -> Option<Vec<EdgeId>> {
        path.into_iter()
            .map(|r| *r.borrow())
            .tuple_windows()
            .map(|(a, b)| self.topo_edge_id(a, b))
            .collect()
    }

    /// Translate a list of paths using `self.path`. The resulting list of paths will only contain
    /// those paths that actually exist.
    pub fn paths<I, J, R>(&self, paths: I) -> Vec<Vec<EdgeId>>
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = R>,
        R: Borrow<RouterId>,
    {
        paths.into_iter().filter_map(|p| self.path(p)).collect()
    }

    /// Get the pair of `RouterId`s that correspond to the given `EdgeId`.
    pub fn net_link(&self, id: impl Borrow<EdgeId>) -> (RouterId, RouterId) {
        let (a, b) = self.graph.edge_endpoints(*id.borrow()).unwrap();
        (self.net_id(a), self.net_id(b))
    }

    /// Transform an EdgeList back to network indices
    pub fn edge_list_to_hashmap<T>(
        &self,
        list: EdgeList<T, TopologyType>,
    ) -> HashMap<(RouterId, RouterId), T> {
        list.into_idx_val().map(|(e, t)| (self.net_link(e), t)).collect()
    }

    /// Iterate over all links (i.e., pairs of edges). Each iteration will yield two edges that have
    /// the same endpoint (but go in opposite direction)
    pub fn links(&self) -> Vec<[EdgeId; 2]> {
        self.graph
            .node_indices()
            .flat_map(|x| self.graph.neighbors(x).map(move |n| (x, n)))
            .filter(|(a, b)| (a < b))
            .map(|(a, b)| {
                [
                    self.graph.find_edge(a, b).unwrap(),
                    self.graph.find_edge(b, a).unwrap(),
                ]
            })
            .collect()
    }

    /// Iterate over all possible combinations of links that contain up to `k` different links.
    pub fn link_combinations(&self, k: usize) -> Vec<Vec<EdgeId>> {
        self.links()
            .into_iter()
            .powerset()
            .take_while(|x| x.len() <= k)
            .map(|x| x.into_iter().flatten().collect::<Vec<EdgeId>>())
            .collect_vec()
    }

    /// Iterate over all possible combinations of edges that contain up to `k` different edges.
    pub fn edge_combinations(&self, k: usize) -> Vec<Vec<EdgeId>> {
        self.graph.edge_indices().powerset().take_while(|x| x.len() <= k).collect_vec()
    }
}
