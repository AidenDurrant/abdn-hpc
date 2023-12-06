#!/bin/bash
#SBATCH --nodes=1 # number of nodes
#SBATCH --cpus-per-task=12 # number of cores
#SBATCH --mem=32G # memory pool for all cores

#SBATCH --ntasks-per-node=1 # one job per node
#SBATCH --gres=gpu:7 # 7 of the 21 partitions
#SBATCH --partition=gpu

#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=<username>@abdn.ac.uk 

module load anaconda3
source activate test

srun python example_script.py --epochs=25 --save $HOME/sharedscratch/
