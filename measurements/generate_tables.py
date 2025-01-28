import pandas as pd
import os
import sys
from datetime import datetime

TABLE1_COLUMNS = [
    (0, 2.0, 3.0, 4.0),
    (0, 2.5, 3.0, 4.0),
    (0, 3.0, 3.0, 4.0),
    (0, 3.5, 3.0, 4.0),
    (0, 4.0, 3.0, 4.0),
    (1, 3.0, 2.0, 4.0),
    (1, 3.0, 2.5, 4.0),
    (1, 3.0, 3.0, 4.0),
    (1, 3.0, 3.5, 4.0),
    (1, 3.0, 4.0, 4.0),
    (2, 3.0, 3.0, 2.0),
    (2, 3.0, 3.0, 2.5),
    (2, 3.0, 3.0, 3.0),
    (2, 3.0, 3.0, 3.5),
    (2, 3.0, 3.0, 4.0),
    (2, 3.0, 3.0, 4.5),
    (2, 3.0, 3.0, 5.0),
]

TABLE2_COLUMNS = [
    (3.0, 3.0, 4.0, 100),
    (3.0, 3.0, 4.0, 300),
    (3.0, 3.0, 4.0, 600),
    (3.0, 3.0, 4.0, 1000),
]


def q0(x):
    return x.min()


def q25(x):
    return x.quantile(0.25)


def q50(x):
    return x.median()


def q75(x):
    return x.quantile(0.75)


def q100(x):
    return x.max()


def mean(x):
    return x.mean()


def is_measurement(f, root):
    if not os.path.isfile(f):
        return False
    if not f.startswith(root):
        return False
    if not f.endswith(".csv"):
        return False
    try:
        datetime.fromisoformat(f[len(root) : -len(".csv")])
    except Exception:
        return False
    return True


def read_most_recent(root_filename):
    filename = sorted(f for f in os.listdir() if is_measurement(f, root_filename))[-1]
    print(f"reading {filename}")
    return pd.read_csv(filename)


def accuracy_table(data):
    df = data
    df = df[df["num_te_paths"] == 0]
    df = df[df["num_externals"] == 30]
    df = df[df["num_initial_states"] == 30]
    df = df[(df["input_changes"] == 100)]
    # df = df[df["topo"] == "TataNld"]

    # select the rows
    df = df[
        [
            "cluster_target",
            "repulsion_target",
            "repulsion_sampled",
            "repulsion_measured",
            "attraction_target",
            "attraction_measured",
            "attraction_sampled",
            "friction_target",
            "friction_measured",
            "friction_sampled",
            "clustering_error",
            "approx_error_max",
            "efficiency_speedup",
        ]
    ]
    df["fraction"] = df["approx_error_max"] / df["clustering_error"]

    df = df.groupby(
        [
            "repulsion_target",
            "attraction_target",
            "friction_target",
            "cluster_target",
        ]
    ).aggregate([q0, q25, q50, q75, q100, mean])
    print(df)

    print("\nTable 1\n-------\n")
    print(
        " rep        | att        | fri        | clustering error (epsilon)                    | approximatoin error (delta)                   | efficiency speedup"
    )
    print(
        "------------+------------+------------+-----------------------------------------------+-----------------------------------------------+----------------------------------------------"
    )
    for column in TABLE1_COLUMNS:
        (i, rep, att, fri) = column
        row = df.loc[(rep, att, fri, 300)]
        clu_err_q00 = row[("clustering_error", "q0")]
        clu_err_q25 = row[("clustering_error", "q25")]
        clu_err_q50 = row[("clustering_error", "q50")]
        clu_err_q75 = row[("clustering_error", "q75")]
        clu_err_q99 = row[("clustering_error", "q100")]
        app_err_q00 = row[("approx_error_max", "q0")]
        app_err_q25 = row[("approx_error_max", "q25")]
        app_err_q50 = row[("approx_error_max", "q50")]
        app_err_q75 = row[("approx_error_max", "q75")]
        app_err_q99 = row[("approx_error_max", "q100")]
        speedup_q00 = row[("efficiency_speedup", "q0")]
        speedup_q25 = row[("efficiency_speedup", "q25")]
        speedup_q50 = row[("efficiency_speedup", "q50")]
        speedup_q75 = row[("efficiency_speedup", "q75")]
        speedup_q99 = row[("efficiency_speedup", "q100")]
        rep_s = row[("repulsion_sampled", "mean")]
        att_s = row[("attraction_sampled", "mean")]
        fri_s = row[("friction_sampled", "mean")]
        rep_m = row[("repulsion_measured", "mean")]
        att_m = row[("attraction_measured", "mean")]
        fri_m = row[("friction_measured", "mean")]

        scenario = f" {rep_m:>3.1f} ({rep_s:>4.2f}) | {att_m:>3.1f} ({att_s:>4.2f}) | {fri_m:>1.1f} ({fri_s:>4.2f})"

        clu = f"[{clu_err_q00:.5f}, {clu_err_q25:.5f}, {clu_err_q50:.5f}, {clu_err_q75:.5f}, {clu_err_q99:.5f}]"
        app = f"[{app_err_q00:.5f}, {app_err_q25:.5f}, {app_err_q50:.5f}, {app_err_q75:.5f}, {app_err_q99:.5f}]"
        spd = f"[{speedup_q00:07.4f}, {speedup_q25:07.4f}, {speedup_q50:07.4f}, {speedup_q75:07.4f}, {speedup_q99:07.4f}]"

        print(f"{scenario} | {clu} | {app} | {spd}")

    print("\nTable 2\n-------\n")
    print(
        "clusters | clustering error (epsilon)                    | approximation error (delta)                   | efficiency speedup"
    )
    print(
        "---------+-----------------------------------------------+-----------------------------------------------+----------------------------------------------"
    )

    for column in TABLE2_COLUMNS:
        (rep, att, fri, clu) = column
        row = df.loc[column]
        clu_err_q00 = row[("clustering_error", "q0")]
        clu_err_q25 = row[("clustering_error", "q25")]
        clu_err_q50 = row[("clustering_error", "q50")]
        clu_err_q75 = row[("clustering_error", "q75")]
        clu_err_q99 = row[("clustering_error", "q100")]
        app_err_q00 = row[("approx_error_max", "q0")]
        app_err_q25 = row[("approx_error_max", "q25")]
        app_err_q50 = row[("approx_error_max", "q50")]
        app_err_q75 = row[("approx_error_max", "q75")]
        app_err_q99 = row[("approx_error_max", "q100")]
        speedup_q00 = row[("efficiency_speedup", "q0")]
        speedup_q25 = row[("efficiency_speedup", "q25")]
        speedup_q50 = row[("efficiency_speedup", "q50")]
        speedup_q75 = row[("efficiency_speedup", "q75")]
        speedup_q99 = row[("efficiency_speedup", "q100")]

        scenario = f"{clu:>8}"
        clu = f"[{clu_err_q00:.5f}, {clu_err_q25:.5f}, {clu_err_q50:.5f}, {clu_err_q75:.5f}, {clu_err_q99:.5f}]"
        app = f"[{app_err_q00:.5f}, {app_err_q25:.5f}, {app_err_q50:.5f}, {app_err_q75:.5f}, {app_err_q99:.5f}]"
        spd = f"[{speedup_q00:07.4f}, {speedup_q25:07.4f}, {speedup_q50:07.4f}, {speedup_q75:07.4f}, {speedup_q99:07.4f}]"
        print(f"{scenario} | {clu} | {app} | {spd}")
    print("\n")


def accuracy_table_latex(data):
    df = data
    df = df[df["num_te_paths"] == 0]
    df = df[df["num_externals"] == 30]
    df = df[df["num_initial_states"] == 30]
    df = df[(df["input_changes"] == 100)]
    # df = df[df["topo"] == "TataNld"]

    # select the rows
    df = df[
        [
            "cluster_target",
            "repulsion_target",
            "repulsion_sampled",
            "repulsion_measured",
            "attraction_target",
            "attraction_measured",
            "attraction_sampled",
            "friction_target",
            "friction_measured",
            "friction_sampled",
            "clustering_error",
            "approx_error_max",
            "efficiency_speedup",
        ]
    ]
    df["fraction"] = df["approx_error_max"] / df["clustering_error"]

    df = df.groupby(
        [
            "repulsion_target",
            "attraction_target",
            "friction_target",
            "cluster_target",
        ]
    ).aggregate([q0, q25, q50, q75, q100, mean])

    print("\nTable 1\n-------\n")
    prev = -1
    for column in TABLE1_COLUMNS:
        (i, rep, att, fri) = column
        row = df.loc[(rep, att, fri, 300)]
        clu_err_q00 = row[("clustering_error", "q0")]
        clu_err_q25 = row[("clustering_error", "q25")]
        clu_err_q50 = row[("clustering_error", "q50")]
        clu_err_q75 = row[("clustering_error", "q75")]
        clu_err_q99 = row[("clustering_error", "q100")]
        app_err_q00 = row[("approx_error_max", "q0")]
        app_err_q25 = row[("approx_error_max", "q25")]
        app_err_q50 = row[("approx_error_max", "q50")]
        app_err_q75 = row[("approx_error_max", "q75")]
        app_err_q99 = row[("approx_error_max", "q100")]
        speedup_q00 = row[("efficiency_speedup", "q0")]
        speedup_q25 = row[("efficiency_speedup", "q25")]
        speedup_q50 = row[("efficiency_speedup", "q50")]
        speedup_q75 = row[("efficiency_speedup", "q75")]
        speedup_q99 = row[("efficiency_speedup", "q100")]
        rep_s = row[("repulsion_sampled", "mean")]
        att_s = row[("attraction_sampled", "mean")]
        fri_s = row[("friction_sampled", "mean")]
        rep_m = row[("repulsion_measured", "mean")]
        att_m = row[("attraction_measured", "mean")]
        fri_m = row[("friction_measured", "mean")]

        scenario = ""
        if i == 0:
            scenario = f"\\textbf{{{rep_m:>4.2f} ({rep_s:>3.1f})}} &   \\WEAK{{{att_m:>4.2f} ({att_s:>3.1f})}} &   \\WEAK{{{fri_m:>4.2f} ({fri_s:>3.1f})}}"
        if i == 1:
            scenario = f"  \\WEAK{{{rep_m:>4.2f} ({rep_s:>3.1f})}} & \\textbf{{{att_m:>4.2f} ({att_s:>3.1f})}} &   \\WEAK{{{fri_m:>4.2f} ({fri_s:>3.1f})}}"
        if i == 2:
            scenario = f"  \\WEAK{{{rep_m:>4.2f} ({rep_s:>3.1f})}} &   \\WEAK{{{att_m:>4.2f} ({att_s:>3.1f})}} & \\textbf{{{fri_m:>4.2f} ({fri_s:>3.1f})}}"

        clu = f"\\EpsilonBoxPlot{{{clu_err_q00:.8f}}}{{{clu_err_q25:.8f}}}{{{clu_err_q50:.8f}}}{{{clu_err_q75:.8f}}}{{{clu_err_q99:.8f}}}"
        app = f"\\DeltaBoxPlot{{{app_err_q00:.8f}}}{{{app_err_q25:.8f}}}{{{app_err_q50:.8f}}}{{{app_err_q75:.8f}}}{{{app_err_q99:.8f}}}"
        spd = f"\\EfficiencyBoxPlot{{{speedup_q00:09.6f}}}{{{speedup_q25:09.6f}}}{{{speedup_q50:09.6f}}}{{{speedup_q75:09.6f}}}{{{speedup_q99:09.6f}}}"

        if prev != i:
            print("\n\\cmidrule(lr){1-7}\n")
        prev = i

        print(f"  yes & {scenario} & {clu} & {app} & {spd} \\\\")

    print("\nTable 2\n-------\n")

    for column in TABLE2_COLUMNS:
        (rep, att, fri, clu) = column
        row = df.loc[column]
        clu_err_q00 = row[("clustering_error", "q0")]
        clu_err_q25 = row[("clustering_error", "q25")]
        clu_err_q50 = row[("clustering_error", "q50")]
        clu_err_q75 = row[("clustering_error", "q75")]
        clu_err_q99 = row[("clustering_error", "q100")]
        app_err_q00 = row[("approx_error_max", "q0")]
        app_err_q25 = row[("approx_error_max", "q25")]
        app_err_q50 = row[("approx_error_max", "q50")]
        app_err_q75 = row[("approx_error_max", "q75")]
        app_err_q99 = row[("approx_error_max", "q100")]
        speedup_q00 = row[("efficiency_speedup", "q0")]
        speedup_q25 = row[("efficiency_speedup", "q25")]
        speedup_q50 = row[("efficiency_speedup", "q50")]
        speedup_q75 = row[("efficiency_speedup", "q75")]
        speedup_q99 = row[("efficiency_speedup", "q100")]
        fraction = row[("fraction", "q50")] * 100

        scenario = f"{clu:>4}"
        clu = f"\\EpsilonBoxPlot{{{clu_err_q00:.8f}}}{{{clu_err_q25:.8f}}}{{{clu_err_q50:.8f}}}{{{clu_err_q75:.8f}}}{{{clu_err_q99:.8f}}}"
        app = f"\\DeltaBoxPlot{{{app_err_q00:.8f}}}{{{app_err_q25:.8f}}}{{{app_err_q50:.8f}}}{{{app_err_q75:.8f}}}{{{app_err_q99:.8f}}}"
        spd = f"\\EfficiencyBoxPlot{{{speedup_q00:09.6f}}}{{{speedup_q25:09.6f}}}{{{speedup_q50:09.6f}}}{{{speedup_q75:09.6f}}}{{{speedup_q99:09.6f}}}"
        print(f"{scenario} & {clu} & {app} & {spd} \\\\")
    print("\n")


def print_usage():
    print(f"Usage: python {sys.argv[0]} [latex]")
    sys.exit(1)


if __name__ == "__main__":

    args = set(sys.argv[1:])
    data = {}
    options = {"latex"}

    for a in args:
        if a not in options:
            print(f"ERROR: Invalid option: {a}\n")
            print_usage()

    data = read_most_recent("accuracy-")

    if "latex" in args:
        accuracy_table_latex(data)
    else:
        accuracy_table(data)
