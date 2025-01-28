//! Module that describes the explorer interface to explore a grid in parallel.

use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering::Relaxed},
        Arc, Mutex,
    },
    thread::sleep,
    time::Duration,
};

use bgpsim::{
    event::BasicEventQueue, network::Network as INetwork, topology_zoo::TopologyZoo,
    types::SimplePrefix,
};
use crossbeam::queue::ArrayQueue as Queue;
use indicatif::{MultiProgress, ProgressBar, ProgressFinish, ProgressStyle};
use itertools::Itertools;
use rand::{thread_rng, Rng};
use serde::Serialize;
use time::{format_description::well_known::Rfc3339, OffsetDateTime};

use crate::scenario::{ScenarioBuilder, ScenarioTopo};

fn progress_template(message: impl std::fmt::Display) -> String {
    format!("{message:60} {{pos:>4}}/{{len:4}}  {{elapsed:>3}}/{{eta:3}} {{msg:>3}} threads [{{wide_bar}}] {{percent:>3}}%")
}

type Network = INetwork<SimplePrefix, BasicEventQueue<SimplePrefix>>;

/// Builder type to create a new explorer
#[derive(Debug)]
pub struct Explorer<Net, Grid> {
    net_params: Vec<Net>,
    grid_params: Vec<Grid>,
    num_inner_params: usize,
    num_threads: usize,
    filename: String,
}

impl<Net, Grid> Default for Explorer<Net, Grid> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Net, Grid> Explorer<Net, Grid> {
    /// Generate a new, empty explorer
    pub fn new() -> Self {
        Self {
            net_params: Vec::new(),
            grid_params: Vec::new(),
            num_inner_params: 1,
            num_threads: rayon::current_num_threads(),
            filename: format!(
                "explore-{}.csv",
                OffsetDateTime::now_utc().format(&Rfc3339).unwrap()
            ),
        }
    }

    /// Set the filename for the output csv
    pub fn filename(mut self, s: impl Into<String>) -> Self {
        self.filename = s.into();
        self
    }

    /// Set the filename for the output csv. The filename will be extended by the current
    /// timestamp. The resulting file name will be: `{s}-{date}.csv`.
    pub fn filename_with_timestamp(mut self, s: impl std::fmt::Display) -> Self {
        self.filename = format!(
            "{}-{}.csv",
            s,
            OffsetDateTime::now_utc().format(&Rfc3339).unwrap()
        );
        self
    }

    /// Set the number of threads (by default, use all threads available, or whatever is set in the
    /// environment variable `RAYON_NUM_THREADS`).
    pub fn num_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Set the different grid parameters for the network.
    pub fn net_params(mut self, params: impl IntoIterator<Item = Net>) -> Self {
        self.net_params = params.into_iter().collect();
        self
    }

    /// Set the grid parameters for the grid search.
    pub fn grid_params(mut self, params: impl IntoIterator<Item = Grid>) -> Self {
        self.grid_params = params.into_iter().collect();
        self
    }

    /// Set the number of inner grid elements.
    pub fn num_inner_iter(mut self, num: usize) -> Self {
        self.num_inner_params = num;
        self
    }
}

impl<Net, Grid> Explorer<Net, Grid>
where
    Net: NetworkSampler + Clone + Send + 'static,
    Grid: Clone + Send + Sync + 'static,
{
    /// Perform the gridsearch experiment. The `outer` function prepares the data for each iteration
    /// of the `inner_params`, while the `inner` function uses the data from `outer` to compute the
    /// actual values for the current grid iteration.
    pub fn work<F, O>(self, func: F)
    where
        F: Copy + Send + 'static + Fn(&Task<Net, Grid>) -> Vec<O>,
        O: Serialize,
    {
        // build the CSV writer
        let writer = csv::WriterBuilder::new()
            .delimiter(b',')
            .has_headers(true)
            .from_path(&self.filename)
            .unwrap();

        let writer = Arc::new(Mutex::new(writer));

        let dispatcher = Arc::new(TaskDispatcher::new(
            self.net_params,
            self.grid_params,
            self.num_inner_params,
        ));

        (0..self.num_threads)
            .map(|_| {
                let writer = writer.clone();
                let dispatcher = dispatcher.clone();
                std::thread::spawn(move || {
                    while let Some(task) = dispatcher.pop() {
                        // increment the thread count count
                        {
                            let mut counter = task.thread_counter.lock().unwrap();
                            *counter += 1;
                            task.pb.set_message(counter.to_string());
                        }
                        let results = func(&task);
                        // write to csv
                        {
                            let mut w = writer.lock().unwrap();
                            for row in results {
                                w.serialize(&row).unwrap();
                                w.flush().unwrap();
                            }
                        }
                        // decrement the thread count
                        {
                            let mut counter = task.thread_counter.lock().unwrap();
                            *counter -= 1;
                            task.pb.set_message(counter.to_string());
                        }
                    }
                })
            })
            .collect_vec()
            .into_iter()
            .for_each(|job| job.join().unwrap());
    }
}

/// A single task to execute.
#[derive(Debug)]
pub struct Task<Net, Grid> {
    /// The network, wrapped in an ARC.
    pub net: Arc<Network>,
    /// The parameters used to sample the network
    pub net_params: Net,
    /// The parameters of the grid.
    pub grid_params: Grid,
    /// The progress bar
    pub pb: ProgressBar,
    /// Counter for the number of threads working on the same network
    thread_counter: Arc<Mutex<usize>>,
}

/// Trait that tries to sample a network.
pub trait NetworkSampler {
    /// Sample a network and its configuration.
    fn sample(&self) -> Option<Network>;

    /// The message to be generated for the progress bar
    fn message(&self) -> String;
}

/// Network parameters (the topology, number of external routers, and the configuration seed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NetParams {
    /// The topology
    pub topo: TopologyZoo,
    /// Number of external routers
    pub num_externals: usize,
    /// Random seed used to generate the config.
    pub config_seed: u64,
}

impl From<(TopologyZoo, usize, u64)> for NetParams {
    fn from(value: (TopologyZoo, usize, u64)) -> Self {
        Self {
            topo: value.0,
            num_externals: value.1,
            config_seed: value.2,
        }
    }
}

impl NetworkSampler for NetParams {
    fn sample(&self) -> Option<Network> {
        ScenarioBuilder::new(ScenarioTopo::TopologyZoo(self.topo))
            .seed(self.config_seed)
            .external_routers(self.num_externals)
            .build()
            .ok()
    }

    fn message(&self) -> String {
        format!(
            "> {} (n={}, e={}, ext={}, s={})",
            self.topo,
            self.topo.num_internals(),
            self.topo.num_internal_edges(),
            self.num_externals,
            self.config_seed,
        )
    }
}

/// Structure to explore individual tasks
#[derive(Debug)]
pub struct TaskDispatcher<Net, Grid> {
    tasks: Queue<Task<Net, Grid>>,
    grid_params: Vec<Grid>,
    network_producer: Arc<Mutex<VecDeque<Net>>>,
    finished: AtomicBool,
    progress: MultiProgress,
    outer_pb: ProgressBar,
    per_topo_problems: u64,
}

impl<Net, Grid> TaskDispatcher<Net, Grid> {
    /// Create a new TaskDispatcher
    pub fn new(
        net_params: impl IntoIterator<Item = Net>,
        grid_params: impl IntoIterator<Item = Grid>,
        sub_problems: usize,
    ) -> Self {
        let grid_params = grid_params.into_iter().collect_vec();
        let progress = MultiProgress::new();
        let networks = net_params.into_iter().collect::<VecDeque<_>>();
        let outer_pb = progress.insert(
            0,
            ProgressBar::new(networks.len() as u64)
                .with_style(
                    ProgressStyle::default_bar()
                        .template(&progress_template("Grid progress"))
                        .unwrap()
                        .progress_chars("##-"),
                )
                .with_message(rayon::current_num_threads().to_string()),
        );
        let network_producer = Arc::new(Mutex::new(networks));
        let per_topo_problems = (grid_params.len() * sub_problems) as u64;
        let tasks = Queue::new(grid_params.len() * 10);

        Self {
            tasks,
            grid_params,
            network_producer,
            finished: AtomicBool::new(false),
            progress,
            outer_pb,
            per_topo_problems,
        }
    }
}

impl<Net, Grid> TaskDispatcher<Net, Grid>
where
    Net: NetworkSampler + Clone,
    Grid: Clone,
{
    /// Pop the next task from the explorer.
    pub fn pop(&self) -> Option<Task<Net, Grid>> {
        // pop a task if there is some.
        if let Some(task) = self.tasks.pop() {
            return Some(task);
        }

        // try to get the lock for the producer
        if let Ok(ref mut mutex) = self.network_producer.try_lock() {
            // generate the new task
            loop {
                let Some(net_params) = mutex.pop_front() else {
                    // nothing left to do!
                    self.finished.store(true, Relaxed);
                    return None;
                };
                self.outer_pb.inc(1);
                // get the network
                if let Some(net) = net_params.sample() {
                    let net = Arc::new(net);
                    // generate a new progress bar
                    let pb = self.progress.insert_from_back(
                        1,
                        ProgressBar::new(self.per_topo_problems)
                            .with_style(
                                ProgressStyle::default_bar()
                                    .template(&progress_template(net_params.message()))
                                    .unwrap()
                                    .progress_chars("##-"),
                            )
                            .with_message("0")
                            .with_finish(ProgressFinish::AndLeave),
                    );
                    pb.inc(0);
                    let thread_count = Arc::new(Mutex::new(0));
                    // generate the tasks
                    let mut tasks = self.grid_params.iter().map(|grid_params| Task {
                        net: net.clone(),
                        net_params: net_params.clone(),
                        grid_params: grid_params.clone(),
                        pb: pb.clone(),
                        thread_counter: thread_count.clone(),
                    });
                    let task = tasks.next().unwrap();
                    for new_task in tasks {
                        assert!(!self.tasks.is_full());
                        let _ = self.tasks.push(new_task);
                    }
                    return Some(task);
                } else {
                    // otherwhise, we simply continue to the next iteration
                    continue;
                }
            }
        } else {
            // someone else is currently generating the tasks. Wait until either we have something
            // new in the task list, or until the finished flag is set to true
            let mut rng = thread_rng();
            loop {
                sleep(Duration::from_millis(rng.gen_range(1..50)));
                // check if there are new tasks
                if let Some(task) = self.tasks.pop() {
                    return Some(task);
                }
                // check if finished is set to true
                if self.finished.load(Relaxed) {
                    return None;
                }
            }
        }
    }
}
