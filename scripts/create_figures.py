import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zoo-dir", help="ZOO_DIR path", type=str, required=True)
    parser.add_argument("--log-dir", help="LOG_DIR path where the algo folder with the experiments is", type=str,
                        required=True)
    parser.add_argument("--store-dir", help="STORE_DIR path", nargs='*', type=str, default=None)
    parser.add_argument("--algo", help="ALGO", type=str, required=True)
    parser.add_argument("--training-env", help="TRAINING_ENV (without version at end)", type=str, required=True)
    parser.add_argument("--version", help="VERSION (v2 WITHOUT dash)", type=str, default="v2")
    parser.add_argument("--avg-window", help="AVG_WINDOW", type=int, required=True)
    parser.add_argument("--add-args", help="ADD_ARGS (additional arguments)", type=str, required=False)
    parser.add_argument("--run-id", help="RUN_ID (ID of last run for single plot) - MUST CONTAIN UNDERSCORE UPFRONT",
                        type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Control verbosity")
    args = parser.parse_args()

    ZOO_DIR = args.zoo_dir
    LOG_DIR = args.log_dir
    STORE_DIR = args.store_dir
    if STORE_DIR is not None:
        assert isinstance(STORE_DIR, list)
        if len(STORE_DIR) > 0:
            buff = ""
            for dir in STORE_DIR:
                buff += f"{dir} "
            buff = buff[:-1]  # cut off last space
            STORE_DIR = buff
        else:
            STORE_DIR = None
    ALGO = args.algo
    TRAINING_ENV = args.training_env
    VERSION = args.version
    AVG_WINDOW = args.avg_window
    ADD_ARGS = args.add_args
    RUN_ID = args.run_id if args.run_id is not None else ""
    VERBOSE = "--verbose" if args.verbose is True else ""

    os.system(f'cd {ZOO_DIR}/scripts/')

    ### Create Training Plots
    os.system('echo "Creating training plots..."')
    os.system(f'python plot_train.py --algo {ALGO} --env {TRAINING_ENV}-{VERSION}{RUN_ID} --y-axis reward '
              f'--exp-folder {LOG_DIR} --store-folder {STORE_DIR} --episode-window {AVG_WINDOW} '
              f'--show-all-experiments --plot-active-components {VERBOSE}')
    os.system('echo "Created plot_train rewards."')
    os.system(f'python plot_train.py --algo {ALGO} --env {TRAINING_ENV}-{VERSION}{RUN_ID} --y-axis success '
              f'--exp-folder {LOG_DIR} --store-folder {STORE_DIR} --episode-window {AVG_WINDOW} '
              f'--show-all-experiments --plot-active-components {VERBOSE}')
    os.system('echo "Created plot_train success."')
    #
    ### Create Evaluation Plots
    os.system('echo "Creating evaluation plots..."')
    os.system(f'python plot_eval.py --algos {ALGO} --env {TRAINING_ENV}-{VERSION}{RUN_ID} --key reward '
              f'--exp-folders {LOG_DIR} --labels {ALGO.upper()} --store-folder {STORE_DIR} --print-n-trials {VERBOSE} '
              f'--show-all-experiments {ADD_ARGS}')
    os.system('echo "Created plot_eval reward."')
    os.system(f'python plot_eval.py --algos {ALGO} --env {TRAINING_ENV}-{VERSION}{RUN_ID} --key success '
              f'--exp-folders {LOG_DIR} --labels {ALGO.upper()} --store-folder {STORE_DIR} --print-n-trials {VERBOSE} '
              f'--show-all-experiments {ADD_ARGS}')
    os.system('echo "Created plot_eval success."')
    #
    # ### Create Plots to compare to other runs
    # os.system(f'python all_plots.py --algos {ALGO} --env {TRAINING_ENV}-{VERSION} --exp-folders {LOG_DIR} --labels "" --key results --store-folder {STORE_DIR} --no-display --print-n-trials --verbose {ADD_ARGS}')
    # os.system(f'python all_plots.py --algos {ALGO} --env {TRAINING_ENV}-{VERSION} --exp-folders {LOG_DIR} --labels "" --key successes --store-folder {STORE_DIR} --no-display --print-n-trials --verbose {ADD_ARGS}')
