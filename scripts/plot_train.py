"""
Plot training reward/success rate
"""
import argparse
import os

import numpy as np
import seaborn
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from stable_baselines3.common.monitor import LoadMonitorResultsError, load_results
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME, ts2xy, window_func

# Activate seaborn
seaborn.set()

parser = argparse.ArgumentParser("Gather results, plot training reward/success")
parser.add_argument("-a", "--algo", help="Algorithm to include", type=str, required=True)
parser.add_argument("-l", "--label", help="Label to add to title", type=str, default=None)
parser.add_argument("-e", "--env", help="Environment(s) to include", nargs="+", type=str, required=True)
parser.add_argument("-f", "--exp-folder", help="Folders to include", type=str, required=True)
parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[6.4, 4.8])
parser.add_argument("--fontsize", help="Font size", type=int, default=14)
parser.add_argument("-max", "--max-timesteps", help="Max number of timesteps to display", type=int)
parser.add_argument("-x", "--x-axis", help="X-axis", choices=["steps", "episodes", "time"], type=str, default="steps")
parser.add_argument("-y", "--y-axis", help="Y-axis", choices=["success", "reward", "length"], type=str,
                    default="reward")
parser.add_argument("-w", "--episode-window", help="Rolling window size", type=int, default=100)
parser.add_argument("-t", "--store_folders", "--store-folder", help="Folder to store figure in.", nargs='*', type=str,
                    default=None)
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Control verbosity")
parser.add_argument("--show-all-experiments",
                    help="Flag to show graphs for ALL experiments of this algo and experiment. "
                         "As a default, only the last run is shown.",
                    dest='show_all', default=False, action='store_true')
parser.add_argument("-s", "--show", help="Flag whether to show plot or not.", default=False, action='store_true')
parser.add_argument("--plot-active-components", help="Flag whether to try to plot active components.", default=False,
                    action='store_true')

args = parser.parse_args()

algo = args.algo
envs = args.env
log_path = os.path.join(args.exp_folder, algo)
label = f" - {args.label}" if args.label is not None else ""

x_axis = {
    "steps": X_TIMESTEPS,
    "episodes": X_EPISODES,
    "time": X_WALLTIME,
}[args.x_axis]
x_label = {
    "steps": "Timesteps",
    "episodes": "Episodes",
    "time": "Walltime (in hours)",
}[args.x_axis]

y_axis = {
    "success": "is_success",
    "reward": "r",
    "length": "l",
}[args.y_axis]
y_label = {
    "success": "Training Success Rate",
    "reward": "Training Episodic Reward",
    "length": "Training Episode Length",
}[args.y_axis]

dirs = []

for env in envs:
    dirs.extend(
        [
            os.path.join(log_path, folder)
            for folder in os.listdir(log_path)
            if (env in folder and os.path.isdir(os.path.join(log_path, folder)))
        ]
    )

dirs = sorted(dirs, reverse=True)

fig = plt.figure(y_label, figsize=args.figsize)
ax1 = fig.add_subplot(1, 1, 1)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

found_components = False
fig.suptitle(f"{y_label}: {algo.upper()}{label}", fontsize=args.fontsize)
ax1.set_xlabel(f"{x_label}", fontsize=args.fontsize)
ax1.set_ylabel(y_label, fontsize=args.fontsize)
for folder in dirs:
    try:
        data_frame = load_results(folder)
    except LoadMonitorResultsError:
        continue
    if args.max_timesteps is not None:
        data_frame = data_frame[data_frame.l.cumsum() <= args.max_timesteps]
    try:
        y = np.array(data_frame[y_axis])
    except KeyError:
        print(f"No data available for {folder}")
        continue
    x, _ = ts2xy(data_frame, x_axis)
    x_comp = x.copy()

    if args.plot_active_components:
        comp_load_success = False
        try:
            y_comp = np.array(data_frame['num_active_components'])
            comp_load_success = True
        except KeyError:
            print(f"No num_active_components available for {folder}")
        if comp_load_success and not found_components:
            ax2 = ax1.twinx()
            max_comp = 3
        if comp_load_success:
            found_components = True
            ax2.plot(x_comp, y_comp, linewidth=2, color='r', label='active components', zorder=5)
            this_max = int(max(y_comp))
            if this_max > max_comp:
                max_comp = this_max

    # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
    if x.shape[0] >= args.episode_window:
        # Compute and plot rolling mean with window of size args.episode_window
        x, y_mean = window_func(x, y, args.episode_window, np.mean)
        ax1.plot(x, y_mean, linewidth=2, label=folder.split("/")[-1], zorder=10)

    # only plot the last experiment
    if not args.show_all:
        break

if args.plot_active_components and found_components:
    ticks = range(0, max_comp + 1, 1)
    ax2.set_yticks(ticks)
    ax2.set_ylim([0, max_comp + 0.1])
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax2.set_ylabel('number of active action components')

ax1.legend(loc='lower right')
fig.tight_layout()

store_folders = args.store_folders
if store_folders is not None and store_folders != 'None':
    if not isinstance(store_folders, list):
        store_folders = [store_folders]
    for store_path in store_folders:
        if os.path.isdir(store_path):
            figure = os.path.join(store_path, f"training_{args.y_axis}")
            plt.savefig(figure)
            if args.verbose:
                print(f"Saved figure {figure}")
        else:
            if args.verbose:
                print(f"Target path does not exist: {store_path}")

if args.show:
    plt.show()
