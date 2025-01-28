import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from datetime import datetime


def print_usage():
    print(f"Usage: python {sys.argv[0]} [OPTIONS]")
    print("")
    print("OPTIONS can be only one of the following:")
    print("   fig4 fig5 fig8")
    sys.exit(1)


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


def generate_fig4():
    data = pd.read_csv("running_time_prepared.csv", engine="pyarrow")
    px.line(
        data,
        x="num_edges",
        y="running_time_sec_q90",
        color="link_failures",
        log_x=True,
        log_y=True,
    ).show()


def generate_fig5():
    data = read_most_recent("qarc-comparison-")
    data["link_failures"] = data["link_failures"].astype(str)
    px.scatter(
        data,
        x="num_edges",
        y="time",
        color="link_failures",
        symbol="link_failures",
        log_y=True,
    ).show()


def generate_fig8():
    data = pd.read_csv("running_time_prepared.csv", engine="pyarrow")
    fig = px.line(
        data,
        x="num_edges",
        y="analysis_time_sec_q90",
        color="link_failures",
        log_x=True,
        log_y=True,
    )
    cldf = data[data["link_failures"] == 2]
    fig.add_traces(
        go.Scatter(
            x=cldf["num_edges"],
            y=cldf["clustering_time_sec_q90"],
            mode="lines",
            name="clustering time",
            # line={"color": "#ff00ff"},
        )
    )
    fig.show()


if __name__ == "__main__":

    args = set(sys.argv[1:])
    data = {}
    options = {
        "fig4",
        "fig5",
        "fig8",
    }

    if len(args) != 1:
        print("ERROR: you must provide exactly one option!\n")
        print_usage()

    for a in args:
        if a not in options:
            print(f"ERROR: Invalid option: {a}\n")
            print_usage()

    if "fig4" in args:
        generate_fig4()

    if "fig5" in args:
        generate_fig5()

    if "fig8" in args:
        generate_fig8()
