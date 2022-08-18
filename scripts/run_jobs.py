"""
Run multiple experiments on a single machine.
"""
import subprocess
import time

import numpy as np

ALGOS = ["tqc"]
ENVS = ["PandaReach-v2"]
LOG_DIR = "/home/adrian/Downloads/testlogs"
SEEDS = [2, 3, 4]
EVAL_FREQ = 5000
N_EVAL_EPISODES = 100
N_TIMESTEPS = 2000  # int(1e6)

processes = []

for algo in ALGOS:
    for env_id in ENVS:
        for seed in SEEDS:
            args = [
                "--algo",
                algo,
                "--env",
                env_id,
                "--seed",
                seed,
                "--eval-episodes",
                N_EVAL_EPISODES,
                "--eval-freq",
                EVAL_FREQ,
                # "--device \"cuda\"",
                "--n-timesteps",
                N_TIMESTEPS,
                # "--hyperparams",
                # f"n_envs:{N_ENVS} gradient_steps:{N_ENVS}",
                # "--num-threads",
                # N_ENVS,
                "--log-folder",
                LOG_DIR,
                # "--verbose",
                # 0
            ]
            args = list(map(str, args))

            print(60*"+")
            p = subprocess.Popen(["python", "train.py"] + args)
            processes.append(p)
            print(60*"-")
            time.sleep(10.0)

while True:
    ps_status = [p.poll() for p in processes]
    if all(ps is not None for ps in ps_status):
        break
    time.sleep(10.0)

time.sleep(1.0)
print ("Finished all jobs")
