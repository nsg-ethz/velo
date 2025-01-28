# Velo: Verifying maximum link loads in a changing world

This is the prototype of Velo, as presented in the paper: "Verifying maximum link loads in a changing world" by Tibor Schneider, Stefano Vissicchio, and Laurent Vanbever.

## Abstract
To meet ever more stringent requirements, network operators often need to reason about worst-case link loads. Doing so involves analyzing traffic forwarding after failures and BGP route changes. State-of-the-art systems identify failure scenarios causing congestion, but they ignore route changes.

We present Velo, the first verification system that efficiently finds maximum link loads under failures and route changes. The key building block of Velo is its ability to massively reduce the gigantic space of possible route changes thanks to (i) a router-based abstraction for route changes, (ii) a theoretical characterization of scenarios leading to worst-case link loads, and (iii) an approximation of input traffic matrices. We fully implement and extensively evaluate Velo. Velo takes only a few minutes to accurately compute all worst-case link loads in large ISP networks. It thus provides operators with critical support to robustify network configurations, improve network management and take business decisions.

# About this Repository

This repository contains the prototype of Velo (formerly known as Viper).
It contains all binaries and scripts to run our evaluation.
Unfortunately, we cannot make the case study public, as it contains confidential information.

## Dependencies

You need to install a current version of the rust compiler toolchain (version 1.77 or greater).
Follow the instructions on [rustup.rs](rustup.rs).

You also need a python (>= 3.10) installation, including `pandas`, `numpy`, and `plotly`.

## Evaluation

To obtain all the results presented in the paper, you need to first generate the raw data, and then pre-process them using a script.

### Running Time (Figure 4, Figure 8, and results in Section 6.1)

The main entry point for measuring Velo's running time is in the file `src/bin/eval_running_time.rs`.
To evaluate the running time, we run Velo on a large number of topologies (and other parameters).\
The running time depends linearly on the number of threads.
We run this script on a server with 96 threads.

**Warning**: This can take multiple days to complete.
To speed things up, disable the largest network (`Kdl`) of TopologyZoo.
To do just that, you need to modify the file `src/bin/eval_running_time.rs` and add the following line after line 87:

```rust
                .filter(|t| *t != TopologyZoo::Kdl)
```

To obtain all data, run the following:

```sh
cargo run --release --bin=eval_running_time  # writes measurements/running_time-YYYY-MM-DDTHH:MM:SS.MILLISZ.csv
```

To prepare the data, run the following script:

```sh
cd measurements
python process_data.py fig4 # writes measurements/effect_kl_ecdf.csv
cd ..
``` 

To show Figure 4 or Figure 8, run:

```sh
cd measurements
python generate_plot.py fig4
python generate_plot.py fig8
cd ..
``` 

To print the different factors of route changes, number of clusters, number of external networks, and number of exception paths, run:

```sh
cd measurements
python process_data.py time_factors # writes measurements/effect_kl_ecdf.csv
cd ..
``` 

### Comparison with QARC (Figure 5)

The main entry point for comparing Velo with QARC is at `src/bin/eval_viper_vs_qarc.rs`.
The measurements of QARC were obtained from their original paper ["Detecting network load violations for distributed control planes"](https://doi.org/10.1145/3385412.3385976) by inspecting their vector graphics.
The following explains how to obtain the measurements of Velo that compare against QARC.

```sh
cargo run --release --bin=eval_viper_vs_qarc  # writes measurements/qarc-comparison-YYYY-MM-DDTHH:MM:SS.MILLISZ.csv
```

There is no need to pre-process the data. Simply generate the plot as follows:

```sh
cd measurements
python generate_plot.py fig5
cd ..
``` 


### Accuracy (Table 1, Figure 6)

The main entry point for measuring Velo's accuracy is at `src/bin/eval_accuracy.rs`.
To measure Velo's accuracy, we find the worst-case for both the original and the approximated traffic matrix and compare their results.
We iterate over many different parameters for sampling the traffic matrix to find how they affect Velo's accuracy.
We do so for 6 of the largest topologies and run the experiment for each topology 20 times with different seeds to sample the matrix.

**Warning**: This can take multiple days to complete.
To speed things up, reduce the number of seeds to iterate over.
To do just that, you need to modify the file `src/bin/eval_accuracy.rs` and remove some seeds on line 100.


To run the experiment, execute:

```sh
cargo run --release --bin=eval_accuracy # writes measurements/accuracy-YYYY-MM-DDTHH:MM:SS.MILLISZ.csv
```

To generate the data for Tables 1 and 2, run the following:


```sh
cd measurements
python generate_tables.py
cd ..
``` 

This command will print the two tables to standard out.
The five values inside square brackets represent: `[min, q25, q50, q75, max]`.
The box plots in the paper are generated using those five values.

