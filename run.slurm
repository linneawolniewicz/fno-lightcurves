#!/bin/bash
# See https://slurm.schedmd.com/job_array.html
# See https://uhawaii.atlassian.net/wiki/spaces/HPC/pages/430407770/The+Basics+Partition+Information+on+Koa for KOA partition info

#SBATCH --partition=koa # koa, sadow, gpu, shared, kill-shared, exclusive-long
#SBATCH --account=koa
##SBATCH --nodelist=gpu-0008

#BATCH --gres=gpu:1
#SBATCH --cpus-per-task=6 ## 5 cores per task
#SBATCH --mem=36gb ## max amount of memory per node you require
#SBATCH --time=30-00:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared, 7day max on exclusive-long, 14day max on sadow, 30day max on koa

#SBATCH --job-name=fno 
#SBATCH --output=job.out
#SBATCH --output=logs/slurm_output/job-%A.out
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=linneamw@hawaii.edu

# Load python profile, then call python script passing SLURM_ARRAY_TASK_ID as an argument.
source ~/profiles/auto.profile
source activate fno

# use this command to run a python script
# python fno_1d_classification.py 

# use this command to run an ipynb and save outputs in the notebook
# jupyter nbconvert --execute --clear-output fno_1d_classification.ipynb 

# Another command to create a .py script, then run that from a ipynb:
# jupyter nbconvert fno_signal_classification.ipynb --to python
python scripts/fno_signal_classification.py
