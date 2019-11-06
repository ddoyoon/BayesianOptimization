import os
import csv
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go

# import seaborn as sns
import datetime
from collections import defaultdict

import argparse
import ast

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--name", "-n", type=str, required=True, help="name(path) of log"
)
parser.add_argument(
    "--csv", action="store_true", help="whether to create csv file"
)
parser.add_argument(
    "--parallel",
    "-p",
    action="store_true",
    help="whether to plot parallel coordinates view for hyper-parameters",
)
parser.add_argument(
    "--scatter",
    "-s",
    action="store_true",
    help="whether to plot create scatter plot for hyper-parameters",
)
parser.add_argument(
    "--acc", action="store_true", help="whether to plot accuracy"
)


def curr_time():
    return datetime.datetime.now().strftime("%Y-%m-%d")


def parse_train(train_log):
    def is_config_line(line):
        """Predicate: determine if line has train log"""
        return "[CONFIG]" in line

    def is_result_line(line):
        """Predicate: determine if line has train log"""
        return "[RESULT]" in line

    def parse_config_line(line):
        """parse line"""
        config_id = line.split("ID=")[1].split(" ")[0]
        config = ast.literal_eval(line.split("config=")[1])

        return config_id, config

    def parse_result_line(line):
        """parse line"""
        config_id = line.split("ID=")[1].split(" ")[0]
        epoch = int(line.split("iter=")[1].split(" ")[0])
        result = ast.literal_eval(line.split("result=")[1])

        return config_id, epoch, result

    log = defaultdict(dict)

    with open(train_log, "r") as f:
        for line in f:
            if is_config_line(line):
                config_id, config = parse_config_line(line)
                if config_id in log.keys():
                    print("Duplicate experiment id.")
                log[config_id]["config"] = config
                log[config_id]["result"] = defaultdict(dict)
                log[config_id]["trial_id"] = "NaN"
                log[config_id]["final_epoch"] = -1
                log[config_id]["current_best"] = {"epoch": -1, "accuracy": 0}

            elif is_result_line(line):
                config_id, epoch, result = parse_result_line(line)
                log[config_id]["result"][epoch] = result
                if (
                    result["accuracy"]
                    > log[config_id]["current_best"]["accuracy"]
                ):
                    log[config_id]["current_best"] = {
                        "epoch": epoch,
                        "accuracy": result["accuracy"],
                    }
                log[config_id]["final_epoch"] = epoch

    return log


def log_to_csv(csv_file_name, train_dict, global_best, best_so_far):
    fields = [
        "trial_id",
        "id",
        # "num_layers",
        # "eval_batch_size",
        # "sync",
        # "num_inter_threads",
        # "data_format",
        "train_batch_size",
        "momentum",
        "weight_decay",
        "batch_norm_decay",
        "batch_norm_epsilon",
        "learning_rate",
        "optimizer",
        "final_epoch",
        "final_accuracy",
        "best_accuracy",
        "global_best",
    ]

    with open(csv_file_name, "w") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for k, d in sorted(train_dict.items()):
            row_dict = {x: d["config"][x] for x in d["config"] if x in fields}
            row_dict["id"] = k

            # Currently running trials' results will not be written in the json file
            row_dict["trial_id"] = d["trial_id"]
            row_dict["final_epoch"] = d["final_epoch"] + 1  # Counts from 0
            row_dict["final_accuracy"] = (
                d["result"][d["final_epoch"]]["accuracy"]
                if row_dict["final_epoch"] > 0
                else -1
            )
            row_dict["best_accuracy"] = d["current_best"]["accuracy"]
            row_dict["global_best"] = (
                global_best[d["trial_id"]]
                if d["trial_id"] in global_best
                else best_so_far
            )
            w.writerow(row_dict)


def parallel_coord(csv_file):
    df = pd.read_csv(csv_file)

    # Handling categorical parameters
    opt_cat_type = CategoricalDtype(
        categories=[
            "momentum",
            "adam",
            "adagrad",
            "adadelta",
            "sgd",
            "rmsprop",
        ],
        ordered=True,
    )
    df["optimizer"] = df.optimizer.astype(opt_cat_type).cat.codes

    opt_bs_type = CategoricalDtype(
        categories=[32, 64, 128, 256, 512, 1024], ordered=True
    )
    df["train_batch_size"] = df.train_batch_size.astype(opt_bs_type).cat.codes

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df["best_accuracy"],
                colorscale="Portland",
                showscale=True,
                cmin=0.7,
                cmax=1,
            ),
            dimensions=list(
                [
                    dict(
                        tickvals=[0, 1, 2, 3, 4, 5],
                        ticktext=["32", "64", "128", "256", "512", "1024"],
                        label="Batch Size",
                        values=df["train_batch_size"],
                    ),
                    dict(
                        range=[0.4, 0.95],
                        label="Momentum",
                        values=df["momentum"],
                    ),
                    dict(
                        range=[1e-4, 2e-4],
                        label="Weight Decay",
                        values=df["weight_decay"],
                    ),
                    dict(
                        range=[0.8, 0.999],
                        label="Batch Norm Decay",
                        values=df["batch_norm_decay"],
                    ),
                    dict(
                        range=[1e-5, 0.001],
                        label="Batch Norm Epsilon",
                        values=df["batch_norm_epsilon"],
                    ),
                    dict(
                        range=[0.01, 0.11],
                        label="Learning Rate",
                        values=df["learning_rate"],
                    ),
                    dict(
                        tickvals=[0, 1, 2, 3, 4, 5],
                        ticktext=[
                            "momentum",
                            "adam",
                            "adagrad",
                            "adadelta",
                            "sgd",
                            "rmsprop",
                        ],
                        label="Optimizer",
                        values=df["optimizer"],
                    ),
                ]
            ),
        )
    )
    fig.show()


def scatter(idx, train_dict):
    filename = f"./{curr_time()}.pdf"
    with PdfPages(filename) as pdf:

        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        l = np.array(list(idx.keys()))

        axs[0].plot(
            l,
            [
                train_dict[x]["config"]["train_batch_size"]
                for x in idx.values()
            ],
            "o",
            markerSize=2,
        )
        axs[0].set_title("batch size")
        axs[0].set_xlabel("Trial Index")
        fig.suptitle("Hyper-parameter Values by Trial Index", fontsize=16)

        axs[1].plot(
            l,
            [train_dict[x]["config"]["momentum"] for x in idx.values()],
            "o",
            markerSize=2,
        )
        axs[1].set_title("momentum")
        axs[1].set_xlabel("Trial Index")

        pdf.savefig()
        plt.close()

        fig, axs = plt.subplots(1, 2, constrained_layout=True)

        axs[0].plot(
            l,
            [train_dict[x]["config"]["weight_decay"] for x in idx.values()],
            "o",
            markerSize=2,
        )
        axs[0].set_title("weight decay")
        axs[0].set_xlabel("Trial Index")
        fig.suptitle("Hyper-parameter Values by Trial Index", fontsize=16)

        axs[1].plot(
            l,
            [
                train_dict[x]["config"]["batch_norm_decay"]
                for x in idx.values()
            ],
            "o",
            markerSize=2,
        )
        axs[1].set_title("batch norm decay")
        axs[1].set_xlabel("Trial Index")

        pdf.savefig()
        plt.close()

        fig, axs = plt.subplots(1, 2, constrained_layout=True)

        axs[0].plot(
            l,
            [
                train_dict[x]["config"]["batch_norm_epsilon"]
                for x in idx.values()
            ],
            "o",
            markerSize=2,
        )
        axs[0].set_title("batch norm epsilon")
        axs[0].set_xlabel("Trial Index")
        fig.suptitle("Hyper-parameter Values by Trial Index", fontsize=16)

        axs[1].plot(
            l,
            [train_dict[x]["config"]["learning_rate"] for x in idx.values()],
            "o",
            markerSize=2,
        )
        axs[1].set_title("learning rate")
        axs[1].set_xlabel("Trial Index")

        pdf.savefig()
        plt.close()

        fig, axs = plt.subplots(1, 1, constrained_layout=True)

        axs.plot(
            l,
            [train_dict[x]["config"]["optimizer"] for x in idx.values()],
            "o",
            markerSize=2,
        )
        axs.set_title("optimizer")
        axs.set_xlabel("Trial Index")
        fig.suptitle("Hyper-parameter Values by Trial Index", fontsize=16)

        pdf.savefig()
        plt.close()


def plot_accuracy(global_best):
    plt.plot(list(global_best.keys()), list(global_best.values()), "o-", lineWidth=2, markerSize=2)
    plt.title("Global Best Accuracy", fontsize=16)
    plt.xlabel('Trial Index')
    plt.ylabel('Validation Accuracy')
    plt.savefig(f"./{curr_time()}.png")


def __main__(
    experiment_name, acc, create_csv, create_parallel, create_scatter
):
    # Read hyper-parameter values and accuracy per epoch for each experiment
    # Merge outputs from all nodes
    nodes = ["06", "10"]
    train_dict = {}
    for node in nodes:
        train_log = f"./{experiment_name}_{node}.log"
        train_data = parse_train(train_log)
        train_dict.update(train_data)

    # Read in trial index from ray's log
    idx = {}
    with open(
        "./experiment_state-2019-11-01_18-02-54.json"
    ) as experiment_state:
        data = json.load(experiment_state)
        for d in data["checkpoints"]:
            trial_id = int(d["experiment_tag"].split("_")[0])
            if "experiment_id" in d["last_result"]:
                experiment_id = d["last_result"]["experiment_id"]
                train_dict[experiment_id]["trial_id"] = trial_id
                idx[trial_id] = experiment_id

    best_so_far = 0
    global_best = {}
    for trial_id, experiment_id in idx.items():
        if train_dict[experiment_id]["current_best"]["accuracy"] > best_so_far:
            best_so_far = train_dict[experiment_id]["current_best"]["accuracy"]
        global_best[trial_id] = best_so_far

    filename = f"./{curr_time()}.csv"

    if create_csv:
        log_to_csv(filename, train_dict, global_best, best_so_far)
    if create_parallel:
        parallel_coord(filename)
    if create_scatter:
        scatter(idx, train_dict)
    if acc:
        plot_accuracy(global_best)


if __name__ == "__main__":
    args = parser.parse_args()
    experiment_name = args.name
    acc = args.acc
    create_csv = args.csv
    create_parallel = args.parallel
    create_scatter = args.scatter

    __main__(experiment_name, acc, create_csv, create_parallel, create_scatter)
    # TODO: add sorting by index and accuracy into two tabs by default
