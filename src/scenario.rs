//! Module to create scenarios on the fly

use anyhow::Context;
use bgpsim::{
    builder::*,
    prelude::{BasicEventQueue, Network},
    topology_zoo::TopologyZoo,
    types::SimplePrefix,
};
use rand::prelude::*;

/// The topology description for the scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScenarioTopo {
    /// A topology from TopologyZoo
    TopologyZoo(TopologyZoo),
    /// A complete graph
    Complete(usize),
}

/// Builder pattern to create a scenario
pub struct ScenarioBuilder {
    /// The chosen topology
    topo: ScenarioTopo,
    /// Number of external routers
    num_external: usize,
    /// Seed for the RNG
    seed: Option<u64>,
}

impl ScenarioBuilder {
    /// Create a new scenario builder with the following default values:
    ///
    /// - 3 external routers
    /// - No route reflectors (iBGP full-mesh)
    /// - 10 distinct community values
    /// - Identity transfer functions for internal BGP sessions
    /// - Identity transfer functions for external BGP sessions
    /// - 30% chance to change a local-pref value
    /// - If the local-pref value is changed, it is set to either 50 or 150.
    /// - 50% chance to set 1 community in a transfer function, and 50% change to set two.
    /// - Randomized seed.
    pub fn new(topo: ScenarioTopo) -> Self {
        Self {
            topo,
            num_external: 3,
            seed: None,
        }
    }

    /// Set the number of external routers to generate. The default value is 3.
    pub fn external_routers(&mut self, num: usize) -> &mut Self {
        self.num_external = num;
        self
    }

    /// Set the random seed for the generation
    pub fn seed(&mut self, seed: u64) -> &mut Self {
        self.seed = Some(seed);
        self
    }

    /// Build the scenario, returning both the network and the BGP graph.
    pub fn build(&self) -> anyhow::Result<Network<SimplePrefix, BasicEventQueue<SimplePrefix>>> {
        let mut net: Network<SimplePrefix, _> = match self.topo {
            ScenarioTopo::TopologyZoo(topo) => topo.build(BasicEventQueue::default()),
            ScenarioTopo::Complete(n) => {
                Network::build_complete_graph(BasicEventQueue::default(), n)
            }
        };
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        net.build_external_routers(
            extend_to_k_external_routers_seeded,
            (&mut rng, self.num_external),
        )
        .context("Create external routers")?;
        net.build_link_weights_seeded(&mut rng, uniform_integer_link_weight_seeded, (10, 100))
            .context("set link weights")?;
        net.build_ebgp_sessions().context("Establish eBGP sessions")?;
        net.build_ibgp_full_mesh().context("Create iBGP full-mesh")?;

        Ok(net)
    }
}

impl std::fmt::Debug for ScenarioBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScenarioBuilder").finish()
    }
}
