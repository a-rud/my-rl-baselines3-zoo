"""
Run multiple experiments with train.py on a single machine.
"""
import sys
import subprocess
import time
import argparse
import torch as th

from utils.utils import StoreDict

VARYING_KEYS = ['algo', 'env', 'seed', 'device', 'num_trainings']
DICT_KEYS = ['hyperparams', 'env_kwargs']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Do NOT use default values if not necessary (these will be set in the train.py subprocess) (or set default=None)
    # Do NOT specify abbreviations for arguments (e.g. -f for --log-folder)
    parser.add_argument("--algo", help="RL Algorithm", default="tqc", type=str, required=False, nargs='+')
    parser.add_argument("--env", type=str, nargs='+', required=True, help="environment ID")
    parser.add_argument("--num-trainings", type=int, default=1, help="Number of training processes to start")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument("--seed", help="First random generator seed. Will be incremented for resp. trainings if seed>0",
                        type=int)
    parser.add_argument("--seed-offset", type=int, default=1,
                        help="Offset between seeds. Seeds are generated like this: seed_i = seed + i*seed_offset")
    parser.add_argument("--yaml-file", type=str, default=None,
                        help="Custom yaml file from which the hyperparameters will be loaded")
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", type=int)
    parser.add_argument("--n-timesteps", help="Overwrite the number of timesteps", type=int)
    parser.add_argument("--log-folder", help="Log folder", type=str)
    parser.add_argument("--tensorboard-log", help="Tensorboard log dir", type=str)
    parser.add_argument("--log-interval", help="Override log interval (default: -1, no change)", type=int)
    parser.add_argument("--vec-env", help="VecEnv type", type=str, choices=["dummy", "subproc"])
    parser.add_argument("--eval-freq", type=int, help="Evaluate the agent every n steps (if negative, no evaluation). "
                                                      "During hyperparameter optimization n-evaluations is used instead")
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", type=int)
    parser.add_argument("--n-eval-envs", help="Number of environments for evaluation", type=int)
    parser.add_argument("--save-freq", help="Save the model every n steps (if negative, no checkpoint)", type=int)
    parser.add_argument("--hyperparams", type=str, nargs="+", action=StoreDict,
                        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict,
                        help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("--study-name", help="Study name for distributed optimization", type=str)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", type=int)
    parser.add_argument("--delay", help="Delay between starting subprocesses in seconds.", type=float, default=10.)
    parser.add_argument("--poll-period", help="Period in seconds between polling subprocesses to check whether they"
                                              "have finished yet.", type=float, default=30.)
    args = parser.parse_args()

    delay = args.delay
    del args.delay
    poll_period = args.poll_period
    del args.poll_period
    assert isinstance(delay, float) and delay > 0, "Delay must be a positive float number."
    assert isinstance(poll_period, float) and poll_period > 0, "Poll period must be a positive float number."

    # Get lists to iterate through
    ALGOS = args.algo
    if not isinstance(ALGOS, list):
        ALGOS = [ALGOS]
    ENVS = args.env
    if not isinstance(ENVS, list):
        ENVS = [ENVS]
    assert args.num_trainings > 0, "At least one training must be started."
    SEEDS = []
    seed_offset = max(1, args.seed_offset)
    del args.seed_offset
    if args.seed is None:
        SEEDS = [-1 for _ in range(args.num_trainings)]
    else:
        SEEDS = [int(args.seed + i * seed_offset) for i in range(args.num_trainings)]

    # Determine device
    NUM_GPU = 0
    if not th.cuda.is_available() or args.device == "cpu":
        device = "cpu"
    else:
        # if auto use all available GPUs
        if args.device == "auto":
            NUM_GPU = th.cuda.device_count()
            device = "cuda"
        # otherwise use specified device
        else:
            device = args.device
    sys.stdout.write(f"Using device '{device}' with {NUM_GPU} GPUs.\n")
    sys.stdout.flush()
    # create a list of parsed arguments which are constant for all subprocesses.
    const_args = []

    # store the rest of the kwargs
    for key, value in args.__dict__.items():
        # do not save the following keys as they vary between runs
        if key not in VARYING_KEYS:
            # Only store passed arguments
            if value is not None:
                key_str = '--' + key.replace('_', '-')
                const_args.append(key_str)
                # Special case: Dict kwargs. Unwrap the kwargs dictionaries into single strings again
                if key in DICT_KEYS:
                    for k, v in args.__getattribute__(key).items():
                        const_args.append(f"{k}:{v}")
                else:
                    const_args.append(value)

    processes = []
    gpu_idx = 0

    for algo in ALGOS:
        for env_id in ENVS:
            for seed in SEEDS:
                # Alternate between GPUs to use
                if NUM_GPU > 1:
                    gpu_idx = len(processes) % NUM_GPU
                    device = f"cuda:{gpu_idx}"
                cmd_args = [
                    "--algo",
                    algo,
                    "--env",
                    env_id,
                    "--seed",
                    seed,
                    "--device",
                    device
                ]
                for arg in const_args:
                    cmd_args.append(arg)
                cmd_args = list(map(str, cmd_args))

                sys.stdout.write(40 * "+")
                sys.stdout.write("\n")
                sys.stdout.write(f"Starting process #{len(processes) + 1} ...\n")
                sys.stdout.flush()
                p = subprocess.Popen(["python", "train.py"] + cmd_args)
                processes.append(p)
                sys.stdout.write(f"Process #{len(processes)} started with PID {p.pid}.\n")
                sys.stdout.flush()
                time.sleep(delay)

    sys.stdout.write(75 * "*")
    sys.stdout.write("\n")
    sys.stdout.write(f"{len(processes)} processes were started. Waiting for all processes to finish...\n")
    sys.stdout.write(75 * "*")
    sys.stdout.write("\n")
    sys.stdout.flush()
    while True:
        time.sleep(poll_period)
        ps_status = [p.poll() for p in processes]
        if all(ps is not None for ps in ps_status):
            break

    time.sleep(1.0)
    sys.stdout.write("Finished all jobs.\nThe return codes are:\n")
    for idx, stat in enumerate(ps_status):
        sys.stdout.write(f"Process #{idx}\tPID {processes[idx].pid}:\t{stat}\n")
    sys.stdout.flush()
