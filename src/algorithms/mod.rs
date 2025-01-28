//! Module that contains (mostly graph) algorithms

use std::{borrow::Borrow, iter::repeat_with, marker::PhantomData};

use petgraph::{
    algo::{floyd_warshall, BoundedMeasure, FloatMeasure},
    prelude::*,
    stable_graph::IndexType,
    EdgeType,
};
use rand::prelude::*;

// pub mod apsp_k1;
pub mod dijkstra;
pub mod shortest_path_dag;
pub mod shortest_path_tree;
pub mod topology;

pub use shortest_path_dag::ShortestPathDag;
pub use shortest_path_tree::{DestinationSpt, SourceSpt};
pub use topology::{EdgeId, NodeId, Topology, TopologyType};

/// Compute the APSP for a single graph
pub fn apsp<N, E: BoundedMeasure + Copy, D: EdgeType, Ix: IndexType>(
    g: &Graph<N, E, D, Ix>,
) -> NodeList<NodeList<E, Ix>, Ix> {
    let mut result: NodeList<NodeList<E, Ix>, Ix> = NodeList::new(g);
    floyd_warshall(g, |e| *e.weight())
        .unwrap()
        .into_iter()
        .for_each(|((s, t), w)| result[s][t] = w);
    result
}

/// Compute the shortest-path DAG towrads the given target.
pub fn shortest_path_dag<N, E: BoundedMeasure + Copy, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
    apsp: &NodeList<NodeList<E, Ix>, Ix>,
    target: NodeIndex<Ix>,
) -> Graph<(), EdgeIndex<Ix>, Directed, Ix> {
    let mut dag: Graph<(), EdgeIndex<Ix>, Directed, Ix> =
        graph.filter_map(|_, _| Some(()), |_, _| None);

    for r in graph.node_indices() {
        let cost_r = apsp[r][target];
        if cost_r == E::max() {
            continue;
        }
        for in_edge in graph.edges_directed(r, Direction::Incoming) {
            let n = in_edge.source();
            let cost_n = apsp[n][target];
            if cost_n == E::max() {
                continue;
            };
            let w = *in_edge.weight();
            if (cost_r + w) > cost_n {
                // link is not used
            } else {
                dag.add_edge(n, r, in_edge.id());
            }
        }
    }

    dag
}

/// Unique weight, a wrapper around u128
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UniqueWeight(u128);

#[allow(dead_code)]
impl UniqueWeight {
    const MAX: Self = UniqueWeight(u128::MAX);
    const MIN: Self = UniqueWeight(u128::MIN);

    /// Restore the old (unscaled) value.
    fn restore(&self) -> u64 {
        (self.0 >> 63) as u64
    }

    fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }
}

impl From<UniqueWeight> for u64 {
    fn from(value: UniqueWeight) -> Self {
        value.restore()
    }
}

impl From<u64> for UniqueWeight {
    fn from(value: u64) -> Self {
        let mut rng = thread_rng();
        Self(((value as u128) << 63) + rng.gen_range(0..(1u128 << 48)))
    }
}

impl std::ops::Add<UniqueWeight> for UniqueWeight {
    type Output = Self;

    fn add(self, rhs: UniqueWeight) -> Self::Output {
        Self(self.0.saturating_add(rhs.0))
    }
}

impl std::ops::Sub<UniqueWeight> for UniqueWeight {
    type Output = Self;

    fn sub(self, rhs: UniqueWeight) -> Self::Output {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl FloatMeasure for UniqueWeight {
    fn zero() -> Self {
        Self(0)
    }

    fn infinite() -> Self {
        Self(u128::MAX)
    }
}

impl BoundedMeasure for UniqueWeight {
    fn min() -> Self {
        Self(0)
    }

    fn max() -> Self {
        Self(u128::MAX)
    }

    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        let (x, overflow) = self.0.overflowing_add(rhs.0);
        (Self(x), overflow)
    }
}

/// Modify the link weights slightly such that all link weights are unique.
pub fn unique_shortest_paths<N: Copy, D: EdgeType, Ix: IndexType>(
    g: &Graph<N, u64, D, Ix>,
) -> Graph<N, UniqueWeight, D, Ix> {
    g.map(|_, n| *n, |_, e| UniqueWeight::from(*e))
}

/// Interface to construct new `EdgeList`s or `NodeList`s.
pub trait GraphList<Ix> {
    /// Create a new instance preallocated to the required of the graph.
    fn new<N, E, D: EdgeType>(g: &Graph<N, E, D, Ix>) -> Self;
}

impl<T: Default, Ix> GraphList<Ix> for T {
    fn new<N, E, D: EdgeType>(_g: &Graph<N, E, D, Ix>) -> Self {
        Default::default()
    }
}

/// Datastructure storing type `T` for each edge in a graph.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeList<T, Ix> {
    d: Vec<T>,
    ix: PhantomData<Ix>,
}

/// Datastructure storing type `T` for each node in a graph.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeList<T, Ix> {
    d: Vec<T>,
    ix: PhantomData<Ix>,
}

impl<T, Ix> From<NodeList<T, Ix>> for Vec<T> {
    fn from(value: NodeList<T, Ix>) -> Self {
        value.d
    }
}

impl<T, Ix> From<Vec<T>> for NodeList<T, Ix> {
    fn from(d: Vec<T>) -> Self {
        assert!(!d.is_empty());
        Self { d, ix: PhantomData }
    }
}

impl<T, Ix> GraphList<Ix> for EdgeList<T, Ix>
where
    T: GraphList<Ix>,
    Ix: IndexType + std::fmt::Debug,
{
    fn new<N, E, D: EdgeType>(g: &Graph<N, E, D, Ix>) -> Self {
        Self {
            d: repeat_with(|| T::new(g)).take(g.edge_count()).collect(),
            ix: PhantomData,
        }
    }
}

impl<T, Ix> GraphList<Ix> for NodeList<T, Ix>
where
    T: GraphList<Ix>,
    Ix: IndexType + std::fmt::Debug,
{
    fn new<N, E, D: EdgeType>(g: &Graph<N, E, D, Ix>) -> Self {
        Self {
            d: repeat_with(|| T::new(g)).take(g.node_count()).collect(),
            ix: PhantomData,
        }
    }
}

impl<I, T, Ix> std::ops::Index<I> for EdgeList<T, Ix>
where
    I: Borrow<EdgeIndex<Ix>>,
    Ix: IndexType,
{
    type Output = T;

    fn index(&self, idx: I) -> &Self::Output {
        &self.d[EdgeIndex::<Ix>::index(*idx.borrow())]
    }
}

impl<I, T, Ix> std::ops::IndexMut<I> for EdgeList<T, Ix>
where
    I: Borrow<EdgeIndex<Ix>>,
    Ix: IndexType,
{
    fn index_mut(&mut self, idx: I) -> &mut Self::Output {
        &mut self.d[EdgeIndex::<Ix>::index(*idx.borrow())]
    }
}

impl<T, Ix> IntoIterator for EdgeList<T, Ix> {
    type Item = T;

    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.d.into_iter()
    }
}

impl<'a, T, Ix> IntoIterator for &'a EdgeList<T, Ix> {
    type Item = &'a T;

    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.d.iter()
    }
}

impl<I, T, Ix> std::ops::Index<I> for NodeList<T, Ix>
where
    I: Borrow<NodeIndex<Ix>>,
    Ix: IndexType,
{
    type Output = T;

    fn index(&self, idx: I) -> &Self::Output {
        &self.d[NodeIndex::<Ix>::index(*idx.borrow())]
    }
}

impl<I, T, Ix> std::ops::IndexMut<I> for NodeList<T, Ix>
where
    I: Borrow<NodeIndex<Ix>>,
    Ix: IndexType,
{
    fn index_mut(&mut self, idx: I) -> &mut Self::Output {
        &mut self.d[NodeIndex::<Ix>::index(*idx.borrow())]
    }
}

impl<T, Ix> IntoIterator for NodeList<T, Ix> {
    type Item = T;

    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.d.into_iter()
    }
}

impl<'a, T, Ix> IntoIterator for &'a NodeList<T, Ix> {
    type Item = &'a T;

    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.d.iter()
    }
}

#[allow(dead_code)]
impl<T, Ix> EdgeList<T, Ix> {
    /// Get the number of elements in the list
    pub fn len(&self) -> usize {
        self.d.len()
    }

    /// Returns `true` if the edge list is empty
    pub fn is_empty(&self) -> bool {
        self.d.is_empty()
    }

    /// Transform the current `EdgeList` into another by calling `f` for each edge.
    pub fn map<U, F>(self, mut f: F) -> EdgeList<U, Ix>
    where
        F: FnMut(EdgeIndex<Ix>, T) -> U,
        Ix: IndexType,
    {
        EdgeList {
            d: self.into_idx_val().map(|(i, t)| f(i, t)).collect(),
            ix: PhantomData,
        }
    }

    /// Create a new `EdgeList` form an `other` by calling `f` for each edge.
    pub fn from_other<U, F>(other: &EdgeList<U, Ix>, mut f: F) -> Self
    where
        F: FnMut(EdgeIndex<Ix>, &U) -> T,
        Ix: IndexType,
    {
        Self {
            d: other.idx_iter().map(|(i, u)| f(i, u)).collect(),
            ix: PhantomData,
        }
    }

    /// Create a new `EdgeList` by calling `f` for each edge in the graph.
    pub fn from_fn<N, E, D, F>(graph: &Graph<N, E, D, Ix>, f: F) -> Self
    where
        F: FnMut(EdgeIndex<Ix>) -> T,
        D: EdgeType,
        Ix: IndexType,
    {
        Self {
            d: graph.edge_indices().map(f).collect(),
            ix: PhantomData,
        }
    }

    /// create an iterator over pairs of `EdgeList`s.
    pub fn zip<'a, U>(
        &'a self,
        other: &'a EdgeList<U, Ix>,
    ) -> impl Iterator<Item = (&'a T, &'a U)> {
        self.d.iter().zip(other.d.iter())
    }

    /// create an iterator over pairs of `EdgeList`s, with a mutable reference to elements of `self`
    pub fn zip_mut<'a, 'b, U, F>(&'a mut self, other: &'b EdgeList<U, Ix>, mut f: F)
    where
        F: FnMut(&mut T, &'b U),
    {
        self.d.iter_mut().zip(other.d.iter()).for_each(|(t, u)| f(t, u))
    }

    /// create an iterator over pairs of `EdgeList`s, with a mutable reference to elements of
    /// `self`, and over the owned `other`.
    pub fn zip_mut_move<U, F>(&mut self, other: EdgeList<U, Ix>, mut f: F)
    where
        F: FnMut(&mut T, U),
    {
        self.d.iter_mut().zip(other.d).for_each(|(t, u)| f(t, u))
    }

    /// Iterate over all elements in the `EdgeList`.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.d.iter()
    }

    /// Iterate over mutable references of all elements in the `EdgeList`.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.d.iter_mut()
    }

    /// Iterate over all indices.
    pub fn idx(&self) -> impl Iterator<Item = NodeIndex<Ix>>
    where
        Ix: IndexType,
    {
        (0..self.d.len()).map(NodeIndex::<Ix>::new)
    }

    /// Iterate over all elements in the `EdgeList`, along with the `EdgeIndex`.
    pub fn idx_iter(&self) -> impl Iterator<Item = (EdgeIndex<Ix>, &T)>
    where
        Ix: IndexType,
    {
        self.d.iter().enumerate().map(|(i, t)| (EdgeIndex::<Ix>::new(i), t))
    }

    /// Iterate over all mutable elements in the `EdgeList`, along with the `EdgeIndex`.
    pub fn idx_iter_mut(&mut self) -> impl Iterator<Item = (EdgeIndex<Ix>, &mut T)>
    where
        Ix: IndexType,
    {
        self.d.iter_mut().enumerate().map(|(i, t)| (EdgeIndex::<Ix>::new(i), t))
    }

    /// Turn `self` into an iterator over pairs of `EdgeIndex` and `T`.
    pub fn into_idx_val(self) -> impl Iterator<Item = (EdgeIndex<Ix>, T)>
    where
        Ix: IndexType,
    {
        self.d.into_iter().enumerate().map(|(i, t)| (EdgeIndex::<Ix>::new(i), t))
    }
}

#[allow(dead_code)]
impl<T, Ix> NodeList<T, Ix> {
    /// Get the number of elements in the list
    pub fn len(&self) -> usize {
        self.d.len()
    }

    /// Returns `true` if `self` is empty.
    pub fn is_empty(&self) -> bool {
        self.d.is_empty()
    }

    /// Transform the current `NodeList` into another by calling `f` for each node.
    pub fn map<U, F>(self, mut f: F) -> NodeList<U, Ix>
    where
        F: FnMut(NodeIndex<Ix>, T) -> U,
        Ix: IndexType,
    {
        NodeList {
            d: self.into_idx_val().map(|(i, t)| f(i, t)).collect(),
            ix: PhantomData,
        }
    }

    /// Create a new `NodeList` form an `other` by calling `f` for each node.
    pub fn from_other<U, F>(other: &NodeList<U, Ix>, mut f: F) -> Self
    where
        F: FnMut(NodeIndex<Ix>, &U) -> T,
        Ix: IndexType,
    {
        Self {
            d: other.idx_iter().map(|(i, u)| f(i, u)).collect(),
            ix: PhantomData,
        }
    }

    /// Create a new `NodeList` by calling `f` for each node in the graph.
    pub fn from_fn<N, E, D, F>(graph: &Graph<N, E, D, Ix>, f: F) -> Self
    where
        D: EdgeType,
        F: FnMut(NodeIndex<Ix>) -> T,
        Ix: IndexType,
    {
        Self {
            d: graph.node_indices().map(f).collect(),
            ix: PhantomData,
        }
    }

    /// create an iterator over pairs of `NodeList`s.
    pub fn zip<'a, U>(
        &'a self,
        other: &'a NodeList<U, Ix>,
    ) -> impl Iterator<Item = (&'a T, &'a U)> {
        self.d.iter().zip(other.d.iter())
    }

    /// create an iterator over pairs of `NodeList`s, with a mutable reference to elements of `self`
    pub fn zip_mut<'a, 'b, U, F>(&'a mut self, other: &'b NodeList<U, Ix>, mut f: F)
    where
        F: FnMut(&mut T, &'b U),
    {
        self.d.iter_mut().zip(other.d.iter()).for_each(|(t, u)| f(t, u))
    }

    /// Iterate over all indices.
    pub fn idx(&self) -> impl ExactSizeIterator<Item = NodeIndex<Ix>>
    where
        Ix: IndexType,
    {
        (0..self.d.len()).map(NodeIndex::<Ix>::new)
    }

    /// Iterate over all elements in the `NodeList`.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> {
        self.d.iter()
    }

    /// Iterate over mutable references of all elements in the `NodeList`.
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut T> {
        self.d.iter_mut()
    }

    /// Iterate over all elements in the `NodeList`, along with the `NodeIndex`.
    pub fn idx_iter(&self) -> impl ExactSizeIterator<Item = (NodeIndex<Ix>, &T)>
    where
        Ix: IndexType,
    {
        self.d.iter().enumerate().map(|(i, t)| (NodeIndex::<Ix>::new(i), t))
    }

    /// Iterate over all mutable elements in the `NodeList`, along with the `NodeIndex`.
    pub fn idx_iter_mut(&mut self) -> impl ExactSizeIterator<Item = (NodeIndex<Ix>, &mut T)>
    where
        Ix: IndexType,
    {
        self.d.iter_mut().enumerate().map(|(i, t)| (NodeIndex::<Ix>::new(i), t))
    }

    /// Turn `self` into an iterator over pairs of `NodeIndex` and `T`.
    pub fn into_idx_val(self) -> impl ExactSizeIterator<Item = (NodeIndex<Ix>, T)>
    where
        Ix: IndexType,
    {
        self.d.into_iter().enumerate().map(|(i, t)| (NodeIndex::<Ix>::new(i), t))
    }
}

impl<T, Ix: IndexType> NodeList<NodeList<T, Ix>, Ix> {
    /// Swap the dimensions
    pub fn transpose(self) -> Self {
        let n1 = self.d.len();
        let n2 = self.d.first().map(|x| x.d.len()).unwrap_or(0);
        let mut d: Vec<NodeList<T, Ix>> = std::iter::repeat_with(|| NodeList {
            d: Vec::with_capacity(n1),
            ix: PhantomData,
        })
        .take(n2)
        .collect();
        for v in self.d {
            for (i, t) in v.into_idx_val() {
                d[i.index()].d.push(t);
            }
        }
        Self { d, ix: PhantomData }
    }
}
