#!/bin/bash
SECONDS=0 # Reset timer

python train.py --algo tqc --env PandaPickAndPlace-v2 --n-timesteps 4000 --log-folder /home/adrian/Downloads/traininglogs --seed 2 --vec-env dummy --hyperparams n_envs:8 --num-threads -1 --device cuda --eval-freq 1000 --eval-episodes 10


DURATION=$SECONDS # Stop timer
echo "Training took $(($DURATION / 3600)) hours, $((($DURATION / 60) % 60)) minutes and $(($DURATION % 60)) seconds."

# === SUBPROC ===
# env 1 thread -1 = aborted, way too long
# env 4 thread -1 = 1m22s
# env 4 thread  8 = 1m22s, final fps 56-58
# env 6 thread -1 = 1m04s, final fps 80-82
# env 8 thread -1 = 0m54s, final fps 100-105
# Generally, CPU high for 

# === DUMMY ===
# env 4 thread -1 = 1m23s, final fps 53-55
# env 6 thread -1 = 1m01s, final fps 75-77
# env 8 thread -1 = 0m51s, final fps 90-94
