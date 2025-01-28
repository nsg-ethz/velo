import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime


def q10(x):
    return x.quantile(0.1)


def q50(x):
    return x.median()


def q90(x):
    return x.quantile(0.9)


def is_measurement(f, root):
    if not os.path.isfile(f):
        return False
    if not f.startswith(root):
        return False
    if not f.endswith(".csv"):
        return False
    return True


def read_most_recent(root_filename):
    filename = sorted(f for f in os.listdir() if is_measurement(f, root_filename))[-1]
    print(f"reading {filename}")
    return pd.read_csv(filename, engine="pyarrow")


def running_time(data):
    df = data
    df = df[df["num_externals"] == 30]
    df = df[df["num_prefixes"] == 100_000]
    df = df[df["num_initial_states"] == 30]
    df = df[df["cluster_target"] == 300]
    df = df[df["input_changes"] == 10.0]
    df = df[df["num_te_paths"] == 0]

    # select the rows
    df = df[
        [
            "num_edges",
            "link_failures",
            "running_time_sec",
            "analysis_time_sec",
            "clustering_time_sec",
        ]
    ]

    # compute statistics
    df = df.groupby(["num_edges", "link_failures"], as_index=False).aggregate(
        [q10, q50, q90]
    )
    new_columns = [
        f"{col}_{quantile}" if quantile else col for col, quantile in df.columns
    ]
    df.columns = new_columns
    df = df.reset_index()
    df = df.sort_values("num_edges")

    df.to_csv("running_time_prepared.csv", index=False)
    print("Written running_time_prepared.csv. To plot it, run the following:\n")
    print("    python generate_plot.py fig4")
    print("    python generate_plot.py fig8\n")


def compare_kl(data):
    print("compare_kl")
    df = data
    df = df[df["cap"] > 80]
    n = len(df["edge"].unique())

    # reindex
    df.set_index("edge", inplace=True)

    # compute ecdf
    ecdf = pd.DataFrame({"ecdf": np.arange(1, n + 1) / n})
    for k in df["k"].unique():
        if k == 0:
            continue
        for l in df["l"].unique():
            if l == 0:
                continue
            reference = df[(df["k"] == 0) & (df["l"] == l)]["max"]
            compare = df[(df["k"] == k) & (df["l"] == 0)]["max"]
            fraction = compare / reference
            ecdf[f"fraction-k{k}-l{l}"] = sorted(fraction)
    ecdf.to_csv("compare_kl_ecdf.csv", index=False)


def print_aspect_influence(data):
    print("Time factors when changing dimensions, c.f., Section 6.1")

    df = data
    df = df[df["topo"] == "Cogentco"]
    df = df[df["link_failures"] == 2]

    aspects = {
        "input_changes": (10, [1, 100, np.inf]),
        "cluster_target": (300, [100, 600, 1000]),
        "num_externals": (30, [100]),
        "num_te_paths": (0, [1, 3, 5]),
    }

    df = df[[*aspects, "running_time_sec"]]
    df = df.groupby(list(aspects)).median()

    base = df["running_time_sec"][tuple(d for d, _ in aspects.values())]

    for aspect, (_, values) in aspects.items():
        for val in values:
            time = 0
            key = tuple(val if k == aspect else d for k, (d, _) in aspects.items())
            time = df["running_time_sec"][key]
            print(f"{aspect} = {val}: {time} ({time / base})")


def print_usage():
    print(f"Usage: python {sys.argv[0]} [OPTIONS]")
    print("")
    print("OPTIONS can be one of the following:")
    print("   fig4 fig8 time_factors all")
    sys.exit(1)


if __name__ == "__main__":

    args = set(sys.argv[1:])
    data = {}
    options = {
        "fig4",
        "fig8",
        "time_factors",
        "all",
    }

    if len(args) == 0:
        print("ERROR: you must provide at least one option!\n")
        print_usage()

    for a in args:
        if a not in options:
            print(f"ERROR: Invalid option: {a}\n")
            print_usage()

    if "fig4" in args or "fig8" in args or "all" in args:
        file_root = "running-time-"
        if file_root not in data:
            data[file_root] = read_most_recent(file_root)
        running_time(data[file_root])

    if "time_factors" in args or "all" in args:
        file_root = "running-time-"
        if file_root not in data:
            data[file_root] = read_most_recent(file_root)
        print_aspect_influence(data[file_root])
