#!/bin/bash
ZOO_DIR='/home/adrian/master_ws/sb3-collection/rl-baselines3-zoo'
LOG_DIR='/home/adrian/master_ws/artifacts/'
ALGO='tqc'
TRAINING_ENV='PandaReach'
#RUN_ID='_2'  # MUST contain the underscore up front
VERSION='v2'
STORE_DIR='/home/adrian/Downloads/testplots'

# Plotting variables
AVG_WINDOW=20 # Smoothing window for plot_train
ADD_ARGS='--episode-window 10' # additional arguments for plot_eval and all_plots:

cd ${ZOO_DIR}/scripts/
#
### Get the ID of the latest run
if [[ -z "$RUN_ID" ]]; then
  RUN_ID=_$(python return_highest_run_id.py --log-path $LOG_DIR/$ALGO --env ${TRAINING_ENV}-${VERSION})
fi
echo "Plotting run ${TRAINING_ENV}-${VERSION}${RUN_ID}"
if [[ -z "$STORE_DIR" ]]; then
  STORE_DIR=${LOG_DIR}/${ALGO}/${TRAINING_ENV}-${VERSION}${RUN_ID}/
fi
echo "Storing to directory ${STORE_DIR}"
#
### Create Training Plots
echo "Creating training plots..."
python plot_train.py -a $ALGO -e ${TRAINING_ENV}-${VERSION}${RUN_ID} -y reward -f $LOG_DIR --target-folder ${STORE_DIR} --episode-window $AVG_WINDOW
python plot_train.py -a $ALGO -e ${TRAINING_ENV}-${VERSION}${RUN_ID} -y success -f $LOG_DIR --target-folder ${STORE_DIR} --episode-window $AVG_WINDOW
#
### Create Evaluation Plots
python plot_eval.py --algos $ALGO --env $TRAINING_ENV-${VERSION}${RUN_ID} --exp-folders $LOG_DIR --labels "" --key reward --target-folder ${STORE_DIR} --no-display --print-n-trials --verbose $ADD_ARGS
python plot_eval.py --algos $ALGO --env $TRAINING_ENV-${VERSION}${RUN_ID} --exp-folders $LOG_DIR --labels "" --key success --target-folder ${STORE_DIR} --no-display --print-n-trials --verbose $ADD_ARGS
#
### Create Plots to compare to other runs
python all_plots.py --algos $ALGO --env $TRAINING_ENV-${VERSION} --exp-folders $LOG_DIR --labels "" --key results --target-folder ${STORE_DIR} --no-display --print-n-trials --verbose $ADD_ARGS
python all_plots.py --algos $ALGO --env $TRAINING_ENV-${VERSION} --exp-folders $LOG_DIR --labels "" --key successes --target-folder ${STORE_DIR} --no-display --print-n-trials --verbose $ADD_ARGS
### List results
cd ${STORE_DIR}/
touch "${TRAINING_ENV}-${VERSION}${RUN_ID}"
ls -1A ${STORE_DIR}


#$ cd $LOG_DIR/tqc
#$ ls -Al
#total 0
#drwx------ 2 1000 1000 0 Aug  4 17:14 PandaPickAndPlace-v1_1
#drwx------ 2 1000 1000 0 Aug  4 18:49 PandaPickAndPlace-v1_2
#drwx------ 2 1000 1000 0 Aug  4 18:59 PandaPickAndPlace-v1_3
#drwx------ 2 1000 1000 0 Aug  6 20:54 PandaPickAndPlace-v2_1
#drwx------ 2 1000 1000 0 Aug  6 22:25 PandaPickAndPlace-v2_2
#drwx------ 2 1000 1000 0 Aug  6 22:30 PandaPickAndPlace-v2_3
#drwx------ 2 1000 1000 0 Aug  7 05:09 PandaPickAndPlace-v2_4
#drwx------ 2 1000 1000 0 Aug  7 17:40 PandaPickAndPlace-v2_5
#drwx------ 2 1000 1000 0 Aug  8 15:39 PandaPickAndPlace-v2_6
#drwx------ 2 1000 1000 0 Aug 10 19:37 PandaPickAndPlaceJoints-v2_1
#drwx------ 2 1000 1000 0 Aug 11 16:51 PandaPickAndPlaceJointsDense-v2_1
#drwx------ 2 1000 1000 0 Aug 12 08:18 PandaPickAndPlaceJointsDense-v2_2
#drwx------ 2 1000 1000 0 Aug  4 19:12 PandaReach-v1_1
#drwx------ 2 1000 1000 0 Aug  4 19:26 PandaReach-v1_2
#drwx------ 2 1000 1000 0 Aug  6 22:18 PandaReach-v2_1
#drwx------ 2 1000 1000 0 Aug  7 06:28 PandaReach-v2_2
#drwx------ 2 1000 1000 0 Aug  7 20:18 PandaReach-v2_3
#drwx------ 2 1000 1000 0 Aug  8 16:21 PandaReach-v2_4
#drwx------ 2 1000 1000 0 Aug  9 06:21 PandaReachJoints-v2_1
#drwx------ 2 1000 1000 0 Aug  9 13:56 PandaReachJoints-v2_2
#drwx------ 2 1000 1000 0 Aug  9 17:46 PandaReachJoints-v2_3
#drwx------ 2 1000 1000 0 Aug  9 06:20 PandaReachJointsDense-v2_1
#drwx------ 2 1000 1000 0 Aug  9 17:46 PandaReachJointsDense-v2_2
#drwx------ 2 1000 1000 0 Aug 12 09:26 PandaReachJointsDense-v2_3
#drwx------ 2 1000 1000 0 Aug 12 10:14 PandaReachJointsDense-v2_4
#drwx------ 2 1000 1000 0 Aug 12 11:03 PandaReachJointsDense-v2_5
#drwx------ 2 1000 1000 0 Aug 12 11:50 PandaReachJointsDense-v2_6
#drwx------ 2 1000 1000 0 Aug 12 12:36 PandaReachJointsDense-v2_7
#$ cd $LOG_DIR/sac
#$ ls -Al
#total 0
#drwx------ 2 1000 1000 0 Aug  9 21:32 PandaReachJoints-v2_1
#drwx------ 2 1000 1000 0 Aug 10 01:17 PandaReachJoints-v2_2
#drwx------ 2 1000 1000 0 Aug  9 21:32 PandaReachJointsDense-v2_1
#drwx------ 2 1000 1000 0 Aug 10 01:17 PandaReachJointsDense-v2_2