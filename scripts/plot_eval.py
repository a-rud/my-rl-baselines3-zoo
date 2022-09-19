import argparse
import os
import pickle
from copy import deepcopy

import numpy as np
import pytablewriter
import seaborn
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix

from stable_baselines3.common.results_plotter import window_func

parser = argparse.ArgumentParser("Gather results, plot them and create table")
parser.add_argument("-a", "--algos", help="Algorithms to include", nargs="+", type=str)
parser.add_argument("-e", "--env", help="Environments to include", nargs="+", type=str)
parser.add_argument("-f", "--exp-folders", help="Folders to include", nargs="+", type=str)
parser.add_argument("-l", "--labels", help="Label for each folder", nargs="+", type=str)
parser.add_argument("-t", "--target-folder", help="Folder to store figure in.", type=str, default=None)
parser.add_argument(
    "-k", "--key", help="Key from the `evaluations.npz` file to use to aggregate results "
                        "(e.g. reward, success rate, ...), it is 'results' by default (i.e., the episode reward)",
    default="results", type=str,
)
parser.add_argument("-max", "--max-timesteps", help="Max number of timesteps to display", type=int, default=int(3e6))
parser.add_argument("-min", "--min-timesteps", help="Min number of timesteps to keep a trial", type=int, default=-1)
parser.add_argument("-w", "--episode-window", help="Rolling window size", type=int, default=None)
parser.add_argument("-o", "--output", help="Output filename (pickle file), where to save the post-processed data",
                    type=str)
parser.add_argument(
    "-median", "--median", action="store_true", default=False, help="Display median instead of mean in the table"
)
parser.add_argument("--no-divider", action="store_true", default=False, help="Do not convert x-axis scale.")
parser.add_argument("--no-display", action="store_true", default=False, help="Do not show the plots")
parser.add_argument(
    "-print", "--print-n-trials", action="store_true", default=False, help="Print the number of trial for each result"
)
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Control verbosity")
parser.add_argument("--show-all-experiments",
                    help="Flag to show graphs for ALL experiments of this algo and experiment. "
                         "As a default, only the last run is shown.",
                    dest='show_all', default=False, action='store_true')
args = parser.parse_args()

if "success" in args.key:
    args.key = "successes"
elif "reward" in args.key:
    args.key = "results"
elif "length" in args.key:
    args.key = "ep_lengths"

# Activate seaborn
seaborn.set()
results = {}
post_processed_results = {}

args.algos = [algo.upper() for algo in args.algos]

if args.labels is None:
    args.labels = args.exp_folders

y_label = {
    "successes": "Eval Success Rate",
    "results": "Eval Episodic Reward",
    "ep_lengths": "Eval Training Episode Length",
}[args.key]
fig_suffix = {
    "successes": "success",
    "results": "reward",
    "ep_lengths": "length",
}[args.key]

for env in args.env:  # noqa: C901
    plt.figure(f"Results {env}")
    plt.title(f"{env}", fontsize=14)

    x_label_suffix = ""  # if args.no_million else "(in Million)"
    if not args.no_divider:
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xlabel(f"Timesteps {x_label_suffix}", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    results[env] = {}
    post_processed_results[env] = {}

    for algo in args.algos:
        for folder_idx, exp_folder in enumerate(args.exp_folders):

            log_path = os.path.join(exp_folder, algo.lower())

            if not os.path.isdir(log_path):
                print(f"{log_path} is not a directory. Continuing.")
                continue

            results[env][f"{args.labels[folder_idx]}-{algo}"] = 0.0

            dirs = [
                os.path.join(log_path, d)
                for d in os.listdir(log_path)
                if (env in d and os.path.isdir(os.path.join(log_path, d)))
            ]

            # only plot the last experiment
            if not args.show_all:
                dirs = [sorted(dirs)[-1]]

            max_len = 0
            merged_timesteps, merged_results = [], []
            last_eval = []
            timesteps = np.empty(0)
            for _, dir_ in enumerate(dirs):
                try:
                    log = np.load(os.path.join(dir_, "evaluations.npz"))
                except FileNotFoundError:
                    print("Eval not found for", dir_)
                    continue

                mean_ = np.squeeze(log["results"].mean(axis=1))

                if mean_.shape == ():
                    continue

                if args.verbose:
                    print(f"#{len(merged_results) + 1}: Using data from directory {dir_}")

                max_len = max(max_len, len(mean_))
                if len(log["timesteps"]) >= max_len:
                    timesteps = log["timesteps"]

                # For post-processing
                merged_timesteps.append(log["timesteps"])
                merged_results.append(log[args.key])

                # Truncate the plots
                while timesteps[max_len - 1] > args.max_timesteps:
                    max_len -= 1
                timesteps = timesteps[:max_len]

                if len(log[args.key]) >= max_len:
                    last_eval.append(log[args.key][max_len - 1])
                else:
                    last_eval.append(log[args.key][-1])

            # Merge runs with different eval freq:
            # ex: (100,) eval vs (10,)
            # in that case, downsample (100,) to match the (10,) samples
            # Discard all jobs that are < min_timesteps
            min_trials = []
            if args.min_timesteps > 0:
                min_ = np.inf
                for idx, n_timesteps in enumerate(merged_timesteps):
                    if n_timesteps[-1] >= args.min_timesteps:
                        min_ = min(min_, len(n_timesteps))
                        if len(n_timesteps) == min_:
                            max_len = len(n_timesteps)
                            # Truncate the plots
                            while n_timesteps[max_len - 1] > args.max_timesteps:
                                max_len -= 1
                            timesteps = n_timesteps[:max_len]
                    else:
                        if args.verbose:
                            print(f"Discarding results #{idx + 1} because it's shorter than args.min_timesteps.")

                # Avoid modifying original aggregated results
                merged_results_ = deepcopy(merged_results)
                # Downsample if needed
                for trial_idx, n_timesteps in enumerate(merged_timesteps):
                    # We assume they are the same, or they will be discarded in the next step
                    if len(n_timesteps) == min_ or n_timesteps[-1] < args.min_timesteps:
                        pass
                    else:
                        new_merged_results = []
                        # Nearest neighbour
                        distance_mat = distance_matrix(n_timesteps.reshape(-1, 1), timesteps.reshape(-1, 1))
                        closest_indices = distance_mat.argmin(axis=0)
                        for closest_idx in closest_indices:
                            new_merged_results.append(merged_results_[trial_idx][closest_idx])
                        merged_results[trial_idx] = new_merged_results
                        last_eval[trial_idx] = merged_results_[trial_idx][closest_indices[-1]]

            # Remove incomplete runs
            merged_results_tmp, last_eval_tmp = [], []
            for idx in range(len(merged_results)):
                if len(merged_results[idx]) >= max_len:
                    merged_results_tmp.append(merged_results[idx][:max_len])
                    last_eval_tmp.append(last_eval[idx])
                else:
                    if args.verbose:
                        print(f"Dropping results #{idx + 1} because it's shorter than max_len.")

            merged_results = merged_results_tmp
            last_eval = last_eval_tmp

            # Post-process
            if len(merged_results) > 0:
                # shape: (n_trials, n_eval * n_eval_episodes)
                merged_results = np.array(merged_results)
                n_trials = len(merged_results)
                n_eval = len(timesteps)

                if args.print_n_trials:
                    print(f"Number of trials in {env}-{algo}-{args.labels[folder_idx]}: {n_trials}")

                # reshape to (n_trials, n_eval, n_eval_episodes)
                evaluations = merged_results.reshape((n_trials, n_eval, -1))
                # re-arrange to (n_eval, n_trials, n_eval_episodes)
                evaluations = np.swapaxes(evaluations, 0, 1)
                # (n_eval,)
                mean_ = np.mean(evaluations, axis=(1, 2))
                # (n_eval, n_trials)
                mean_per_eval = np.mean(evaluations, axis=-1)
                # (n_eval,)
                std_ = np.std(mean_per_eval, axis=-1)
                # std: error:
                std_error = std_ / np.sqrt(n_trials)
                # Take last evaluation
                # shape: (n_trials, n_eval_episodes) to (n_trials,)
                last_evals = np.array(last_eval).squeeze().mean(axis=-1)
                # Standard deviation of the mean performance for the last eval
                std_last_eval = np.std(last_evals)
                # Compute standard error
                std_error_last_eval = std_last_eval / np.sqrt(n_trials)

                if args.median:
                    results[env][f"{algo}-{args.labels[folder_idx]}"] = f"{np.median(last_evals):.0f}"
                else:
                    results[env][
                        f"{algo}-{args.labels[folder_idx]}"
                    ] = f"{np.mean(last_evals):.0f} +/- {std_error_last_eval:.0f}"

                # Convert x-axis timesteps scale with a 10**exp divider
                divider = 1.0  # this is now done over matplotlib settings
                # if args.no_divider:
                #     divider = 1.0
                # else:
                #     T = timesteps[-1]
                #     exp = 1
                #     if T > 10:
                #         exp = int(np.floor(np.log10(T)))
                #     divider = float(10 ** exp)

                post_processed_results[env][f"{algo}-{args.labels[folder_idx]}"] = {
                    "timesteps": timesteps,
                    "mean": mean_,
                    "std_error": std_error,
                    "last_evals": last_evals,
                    "std_error_last_eval": std_error_last_eval,
                    "mean_per_eval": mean_per_eval,
                }

                plot_label = f"{algo}"
                if f"{args.labels[folder_idx]}":
                    plot_label += f"-{args.labels[folder_idx]}"

                plt.plot(timesteps / divider, mean_, label=plot_label, linewidth=3)
                plt.fill_between(timesteps / divider, mean_ + std_error, mean_ - std_error, alpha=0.5)

                # Smooth the curves:
                if args.episode_window is not None:
                    # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
                    if timesteps.shape[0] >= args.episode_window:
                        if args.verbose:
                            print(f"Applying smoothing window width {args.episode_window}.")
                        # Copy to not impact the variables
                        timesteps_smooth = timesteps.copy()
                        mean_smooth = mean_.copy()
                        std_error_smooth = std_error.copy()
                        # Compute and plot rolling mean with window of size args.episode_window
                        timesteps_smooth, mean_smooth = window_func(timesteps_smooth, mean_smooth, args.episode_window,
                                                                    np.mean)
                        _, std_error_smooth = window_func(timesteps, std_error_smooth, args.episode_window,
                                                          np.mean)
                        # plt.plot(timesteps_smooth, y_mean, linewidth=2, label=folder.split("/")[-1])
                        plt.plot(timesteps_smooth / divider, mean_smooth, label=f"{plot_label} smoothed", linewidth=3)
                        plt.fill_between(timesteps_smooth / divider, mean_smooth + std_error_smooth,
                                         mean_smooth - std_error_smooth, alpha=0.5)

    plt.legend()

# Markdown Table
writer = pytablewriter.MarkdownTableWriter(max_precision=3)
writer.table_name = "results_table"

headers = ["Environments"]

# One additional row for the subheader
value_matrix = [[] for i in range(len(args.env) + 1)]

headers = ["Environments"]
# Header and sub-header
value_matrix[0].append("")
for algo in args.algos:
    for label in args.labels:
        value_matrix[0].append(label)
        headers.append(algo)

writer.headers = headers

for i, env in enumerate(args.env, start=1):
    value_matrix[i].append(env)
    for algo in args.algos:
        for label in args.labels:
            key = f"{algo}-{label}"
            value_matrix[i].append(f'{results[env].get(key, "0.0 +/- 0.0")}')

writer.value_matrix = value_matrix
writer.write_table()

post_processed_results["results_table"] = {"headers": headers, "value_matrix": value_matrix}

if args.output is not None:
    if args.verbose:
        print(f"Saving to {args.output}.pkl")
    with open(f"{args.output}.pkl", "wb") as file_handler:
        pickle.dump(post_processed_results, file_handler)

store_path = args.target_folder
if store_path is not None:
    if os.path.isdir(store_path):
        figure = os.path.join(store_path, f"evaluation_{fig_suffix}")
        plt.savefig(figure)
        if args.verbose:
            print(f"Saved figure {figure}")
    else:
        if args.verbose:
            print(f"Target path does not exist: {store_path}")

if not args.no_display:
    plt.show()
