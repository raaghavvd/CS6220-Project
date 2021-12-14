#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --mem=5Gb
#SBATCH --time=23:00:00
#SBATCH --array=1-100
#SBATCH --output="inference/infer.%A.%a.out"
#SBATCH --error="inference/infer.%A.%a.err"
module load python/3.8.1
PER_TASK=2000
# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $START_NUM + $PER_TASK ))
 
# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

srun python3 runInference.py --start_num $START_NUM --end_num $END_NUM
