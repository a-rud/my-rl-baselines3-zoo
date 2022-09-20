from typing import Tuple
import argparse
import sys
import glob
import os.path


def get_latest_experiment_folder(log_path: str, env_name: str) -> Tuple[str, str]:
    """
    Same function as in utils/utils.py but it doesn't throw gym warnings.

    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: path to log folder
    :param env_name:
    :return: latest run number
    """
    all_paths = glob.glob(os.path.join(log_path, env_name + "_[0-9]*"))
    assert len(all_paths) > 0, f"No valid folders were found at {log_path}/{env_name}"
    sorted_paths = sorted(all_paths, reverse=True)  # sort in descending order
    exp_path = sorted_paths[0]
    exp_folder = os.path.basename(os.path.normpath(exp_path))
    return exp_path, exp_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log-path", help="Path of logs to look for runs", type=str, required=True)
    parser.add_argument("-e", "--env", help="Name of environment. Must NOT include version!", type=str, required=True)
    args = parser.parse_args()

    _, latest_experiment_folder = get_latest_experiment_folder(args.log_path, args.env)
    sys.stdout.write(f"{latest_experiment_folder}")

    # latest_run_name = f"{args.env}_{latest_run_id}"
    # sys.stdout.write(f"{latest_run_name}")
