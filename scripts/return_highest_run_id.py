import argparse
import sys
import glob
import os.path


def get_latest_run_id(log_path: str, env_name: str) -> int:
    """
    Same function as in utils/utils.py but it doesn't throw gym warnings.

    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: path to log folder
    :param env_name:
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, env_name + "_[0-9]*")):
        run_id = path.split("_")[-1]
        path_without_run_id = path[: -len(run_id) - 1]
        if path_without_run_id.endswith(env_name) and run_id.isdigit() and int(run_id) > max_run_id:
            max_run_id = int(run_id)
    return max_run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log-path", help="Path of logs to look for runs", type=str, required=True)
    parser.add_argument("-e", "--env", help="Name of environment. MUST INCLUDE VERSION!", type=str, required=True)
    args = parser.parse_args()

    if "-v" not in args.env:
        args.env += "-v2"
    latest_run_id = get_latest_run_id(args.log_path, args.env)
    sys.stdout.write(f"{latest_run_id}")

    # latest_run_name = f"{args.env}_{latest_run_id}"
    # sys.stdout.write(f"{latest_run_name}")
